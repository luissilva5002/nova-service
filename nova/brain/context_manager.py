"""Append-only token log for one chat, built so llama.cpp's KV cache is reused.

Rules this module enforces:
  * Every message is converted to token ids exactly once and never re-rendered.
  * Assistant replies are stored as the exact ids the model generated.
  * The system block (with the tool schema) is rendered once per session.
  * Retrieved context is shown to the model for one turn only; it is NOT kept
    in history after commit_turn().
  * No history window. Compaction is signalled (needs_compaction) but not done.

Nothing here runs the model. It only needs a tokenizer and the GGUF chat
template, provided by a Backend (LlamaBackend for the real thing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

_DUMMY = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "u"},
    {"role": "assistant", "content": "a"},
]


class ContextOverflow(Exception):
    """The prompt plus the generation reserve does not fit in n_ctx."""


class TemplateNotPrefixStable(Exception):
    """The chat template does not render messages as a stable, growing prefix."""


class Backend(Protocol):
    def encode(self, text: str) -> list[int]:
        """Tokenize with special tokens parsed (<|im_start|> -> one token), no BOS."""

    def render(
        self, messages: list[dict], tools: Optional[list[dict]], add_generation_prompt: bool
    ) -> str:
        """Render messages with the model's chat template."""


class LlamaBackend:
    """Backend over a llama_cpp.Llama (vocab_only=True is enough)."""

    def __init__(self, llama) -> None:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        self._llama = llama
        template = llama.metadata["tokenizer.chat_template"]
        self._fmt = Jinja2ChatFormatter(
            template=template,
            eos_token=self._token_text(llama.token_eos),
            bos_token=self._token_text(llama.token_bos),
            add_generation_prompt=False,
        )

    def _token_text(self, getter) -> str:
        try:
            token_id = getter()
            if token_id is None or token_id < 0:
                return ""
            return self._llama.detokenize([token_id]).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def encode(self, text: str) -> list[int]:
        return list(self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def render(self, messages, tools, add_generation_prompt) -> str:
        self._fmt.add_generation_prompt = add_generation_prompt
        return self._fmt(messages=messages, tools=tools).prompt

    def decode(self, ids: list[int]) -> str:
        return self._llama.detokenize(ids).decode("utf-8", errors="replace")


@dataclass
class _Turn:
    clean_user: list[int]  # user segment without any retrieved-context block
    segments: list[list[int]]  # segments[0] is the user segment as shown to the model


@dataclass
class LoadReport:
    turns_loaded: int
    turns_dropped: int
    messages_skipped: int
    tokens: int


class ContextManager:
    def __init__(
        self,
        backend: Backend,
        n_ctx: int,
        gen_reserve: int = 512,
        compaction_threshold: float = 0.75,
    ) -> None:
        self._b = backend
        self._n_ctx = n_ctx
        self._gen_reserve = gen_reserve
        self._threshold = compaction_threshold
        self._history: list[int] = []
        self._system_ids: list[int] = []
        self._gen_ids: list[int] = []
        self._suffix_ids: list[int] = []
        self._tools: Optional[list[dict]] = None
        self._suffix_text = ""
        self._wrap: dict[str, tuple[str, str]] = {}
        self.template_mode = "unset"  # "derived" or "chatml-fallback"
        self.template_note = ""  # why the fallback was used, if it was
        self._started = False
        self._turn: Optional[_Turn] = None
        self._history_loaded = False

    # ---------- session ----------

    def start_session(self, system_prompt: str, tool_schemas: Optional[list[dict]] = None) -> None:
        if self._started:
            raise RuntimeError("Session already started; tool schemas are fixed per session.")
        self._tools = list(tool_schemas) if tool_schemas else None

        system_text = self._b.render(
            [{"role": "system", "content": system_prompt}], self._tools, False
        )
        self._system_ids = self._b.encode(system_text)
        self._history = list(self._system_ids)

        # Generation prompt, e.g. "<|im_start|>assistant\n"
        head = _DUMMY[:2]
        without = self._b.render(head, None, False)
        with_gen = self._b.render(head, None, True)
        if not with_gen.startswith(without) or len(with_gen) == len(without):
            raise TemplateNotPrefixStable("Cannot derive the generation prompt from the template.")
        self._gen_ids = self._b.encode(with_gen[len(without):])

        # Text after an assistant reply, plus the wrapper text around user/tool messages.
        self._derive_segments()
        self._suffix_ids = self._b.encode(self._suffix_text)
        self._started = True

    # ---------- turns ----------

    def begin_turn(
        self,
        user_text: str,
        context_block: Optional[str] = None,
        context_position: str = "after",
    ) -> list[int]:
        """Start a turn. Returns the full prompt ids for the first model call."""
        self._require_started()
        if self._turn is not None:
            raise RuntimeError("A turn is already open; commit_turn() or abort_turn() first.")
        if context_position not in ("before", "after"):
            raise ValueError("context_position must be 'before' or 'after'.")
        clean = self._b.encode(self._user_segment(user_text))
        shown = clean
        if context_block:
            if context_position == "before":
                shown_text = f"{context_block}\n\n{user_text}"
            else:
                shown_text = f"{user_text}\n\n{context_block}"
            shown = self._b.encode(self._user_segment(shown_text))
        self._turn = _Turn(clean_user=clean, segments=[shown])
        return self._compose_or_abort()

    def continue_after_tool(self, tool_call_ids: list[int], tool_result_text: str) -> list[int]:
        """Record the model's tool call and its result; return the next prompt ids.

        tool_call_ids are the exact ids the model generated for the tool call.
        The prompt keeps everything from the previous call as a prefix.
        """
        self._require_open_turn()
        call_segment = self._gen_ids + self._close(tool_call_ids)
        tool_segment = self._b.encode(self._tool_segment(tool_result_text))
        self._turn.segments.append(call_segment + tool_segment)
        return self._compose_or_abort()

    def commit_turn(self, assistant_token_ids: list[int]) -> None:
        """Finish the turn. Stores the clean user message and the exact reply ids."""
        self._require_open_turn()
        turn = self._turn
        self._history += turn.clean_user
        for segment in turn.segments[1:]:
            self._history += segment
        self._history += self._gen_ids + self._close(assistant_token_ids)
        self._turn = None

    def append_external_turn(self, user_text: str, assistant_text: str) -> None:
        """Append a user/assistant exchange that was answered outside the model."""
        self._require_started()
        if self._turn is not None:
            raise RuntimeError("Cannot append an external turn while a turn is open.")
        if not user_text.strip() or not assistant_text.strip():
            raise ValueError("External user and assistant text must both be non-empty.")

        user_ids = self._b.encode(self._user_segment(user_text))
        assistant_ids = self._gen_ids + self._close(self._b.encode(assistant_text))
        appended_ids = user_ids + assistant_ids
        if len(self._history) + len(appended_ids) > self._n_ctx - self._gen_reserve:
            raise ContextOverflow(
                f"External turn would exceed history limit "
                f"{self._n_ctx - self._gen_reserve}."
            )
        self._history += appended_ids

    def abort_turn(self) -> None:
        self._turn = None

    @property
    def end_of_turn_id(self) -> int:
        if not self._started:
            raise RuntimeError("Call start_session() first.")
        return self._suffix_ids[0]

    def load_history(self, messages: list[dict], max_fraction: float = 0.5) -> LoadReport:
        """Rebuild a fresh session's token history from persisted user/assistant text."""
        self._require_started()
        if self._turn is not None:
            raise RuntimeError("Cannot load history while a turn is open.")
        if self._history_loaded or self._history != self._system_ids:
            raise RuntimeError("History can only be loaded once into a fresh session.")
        if not 0.0 <= max_fraction <= 1.0:
            raise ValueError("max_fraction must be between 0.0 and 1.0.")
        if not messages:
            self._history_loaded = True
            return LoadReport(
                turns_loaded=0,
                turns_dropped=0,
                messages_skipped=0,
                tokens=0,
            )

        skipped = 0
        turns: list[tuple[bool, list[int]]] = []
        current_turn: Optional[list[int]] = None

        for message in messages:
            if not isinstance(message, dict):
                skipped += 1
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
                skipped += 1
                continue

            if role == "user":
                current_turn = self._b.encode(self._user_segment(content))
                turns.append((True, current_turn))
            else:
                assistant_segment = self._gen_ids + self._close(self._b.encode(content))
                if current_turn is None:
                    turns.append((False, assistant_segment))
                else:
                    current_turn.extend(assistant_segment)

        budget = int(max_fraction * (self._n_ctx - self._gen_reserve))
        kept_turns = list(turns)
        turns_dropped = 0

        def token_total(units: list[tuple[bool, list[int]]]) -> int:
            return len(self._system_ids) + sum(len(ids) for _, ids in units)

        while token_total(kept_turns) > budget and kept_turns:
            is_user_turn, _ = kept_turns.pop(0)
            if is_user_turn:
                turns_dropped += 1
            while kept_turns and not kept_turns[0][0]:
                kept_turns.pop(0)

        if token_total(kept_turns) > budget:
            raise ContextOverflow(
                f"System prompt {len(self._system_ids)} exceeds history budget {budget}."
            )

        self._history = list(self._system_ids)
        for _, ids in kept_turns:
            self._history.extend(ids)
        self._history_loaded = True
        return LoadReport(
            turns_loaded=sum(1 for is_user_turn, _ in kept_turns if is_user_turn),
            turns_dropped=turns_dropped,
            messages_skipped=skipped,
            tokens=len(self._history),
        )

    # ---------- size ----------

    def token_count(self) -> int:
        pending = sum(len(s) for s in self._turn.segments) if self._turn else 0
        return len(self._history) + pending

    def needs_compaction(self) -> bool:
        return self.token_count() >= int(self._threshold * (self._n_ctx - self._gen_reserve))

    @property
    def history_ids(self) -> list[int]:
        return list(self._history)

    @property
    def system_ids(self) -> list[int]:
        return list(self._system_ids)

    # ---------- internals ----------

    def _compose(self) -> list[int]:
        prompt = list(self._history)
        for segment in self._turn.segments:
            prompt += segment
        return prompt + self._gen_ids

    def _compose_or_abort(self) -> list[int]:
        prompt = self._compose()
        if len(prompt) + self._gen_reserve > self._n_ctx:
            self._turn = None
            raise ContextOverflow(
                f"Prompt {len(prompt)} + reserve {self._gen_reserve} exceeds n_ctx {self._n_ctx}."
            )
        return prompt

    def _close(self, ids: list[int]) -> list[int]:
        """ids + end-of-turn tokens, without duplicating an end token already present."""
        ids = list(ids)
        if ids and self._suffix_ids and ids[-1] == self._suffix_ids[0]:
            return ids + self._suffix_ids[1:]
        return ids + self._suffix_ids

    def _derive_segments(self) -> None:
        """Learn how the template wraps user/tool messages, using markers.

        Falls back to fixed ChatML text if the template does not behave as
        expected. The equivalence test (T4) is the authority on correctness.
        """
        m1, m2 = "@@NOVA_A@@", "@@NOVA_B@@"
        base = _DUMMY[:2] + [{"role": "assistant", "content": m1}]
        try:
            last = self._b.render(base, None, False)
            i = last.rfind(m1)
            if i < 0:
                raise TemplateNotPrefixStable("assistant marker missing from render")
            tail = last[i + len(m1):]
            if not tail:
                raise TemplateNotPrefixStable("template adds nothing after an assistant reply")
            wrap: dict[str, tuple[str, str]] = {}
            for role in ("user", "tool"):
                out = self._b.render(base + [{"role": role, "content": m2}], None, False)
                i1, i2 = out.find(m1), out.find(m2)
                if i1 < 0 or i2 < i1:
                    raise TemplateNotPrefixStable(f"{role} marker missing from render")
                after = out[i1 + len(m1):]
                if not after.startswith(tail):
                    raise TemplateNotPrefixStable(
                        f"text after an assistant reply differs when it is last vs followed: "
                        f"last={tail!r} followed={after[: len(tail) + 24]!r}"
                    )
                rest = after[len(tail):]
                j = rest.find(m2)
                wrap[role] = (rest[:j], rest[j + len(m2):])
            self._suffix_text, self._wrap = tail, wrap
            self.template_mode, self.template_note = "derived", ""
        except Exception as exc:  # noqa: BLE001 - any template quirk triggers the fallback
            self._suffix_text = "<|im_end|>\n"
            self._wrap = {
                "user": ("<|im_start|>user\n", "<|im_end|>\n"),
                "tool": ("<|im_start|>user\n<tool_response>\n", "\n</tool_response><|im_end|>\n"),
            }
            self.template_mode = "chatml-fallback"
            self.template_note = f"{type(exc).__name__}: {exc}"

    def _user_segment(self, text: str) -> str:
        pre, post = self._wrap["user"]
        return pre + text + post

    def _tool_segment(self, text: str) -> str:
        pre, post = self._wrap["tool"]
        return pre + text + post

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Call start_session() first.")

    def _require_open_turn(self) -> None:
        self._require_started()
        if self._turn is None:
            raise RuntimeError("No open turn; call begin_turn() first.")