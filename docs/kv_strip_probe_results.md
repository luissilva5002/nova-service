# KV-C-lite probe (experiment 6)

- platform: win32 / AMD64, Python 3.13.7
- llama-cpp-python: 0.3.36
- model: qwen2.5-3b-instruct-q4_k_m.gguf
- settings: n_ctx=4096, n_batch=512, n_threads=4, flash_attn=True, kv_type=f16 (default), max_tokens=48

| turn | prompt | evaluated (strip) | evaluated (no strip) | strip | KV == history | strip vs cold | first diff | control vs cold | first diff | cold evaluated |
|---:|---:|---:|---:|---|---|---|---:|---|---:|---:|
| 1 | 220 | 220 | 220 | stripped | yes | - | - | - | - | - |
| 2 | 292 | 85 | 149 | stripped | yes | diverge early | 13 | diverge early | 26 | 292 |
| 3 | 350 | 81 | 148 | stripped | yes | diverge early | 17 | diverge early | 16 | 350 |
| 4 | 413 | 84 | 146 | stripped | yes | diverge early | 10 | identical | - | 413 |
| 5 | 472 | 86 | 153 | stripped | yes | diverge early | 19 | identical | - | 472 |
| 6 | 513 | 78 | 147 | stripped | yes | diverge early | 26 | diverge early | 25 | 513 |
| 7 | 568 | 80 | 133 | stripped | yes | diverge early | 2 | diverge early | 11 | 568 |
| 8 | 635 | 83 | 146 | stripped | yes | diverge early | 16 | identical | - | 635 |

Mean evaluated tokens per turn from turn 2: 82 with strip, 146 without (64 fewer).

### Replies at temperature 0 (stripped KV, cold context, control without shift)

#### Turn 2
- strip:   Sunlight reaches the side of Earth facing away from the Sun by being reflected off the Moon and other celestial bodies. It also passes through space directly, illuminating the Earth's surface.
- cold:    Sunlight reaches the side of Earth facing away from the Sun by reflecting off the Moon. The Moon, being closer to Earth, reflects sunlight that has already traveled through space and been absorbed and re-emitted by the M
- control: Sunlight reaches the side of Earth facing away from the Sun by reflecting off the Moon. The Moon, being closer to Earth, intercepts sunlight and reflects it back to Earth, creating the lunar phases we see.

#### Turn 3
- strip:   Objects look smaller when they are farther away because the angle at which we see them decreases, making them appear smaller in our field of view. This is described by the small angle approximation in geometry.
- cold:    Objects look smaller when they are farther away because the angle at which we see them decreases. As the distance increases, the same amount of light or energy is spread over a larger area, making the object appear small
- control: Objects look smaller when they are farther away because the angle at which we see them (the angular size) decreases. As the distance increases, the same amount of light is spread over a larger area, making the object app

#### Turn 4
- strip:   Clouds form when warm air rises and cools, causing water vapor in the air to condense into liquid droplets. This process is known as condensation.
- cold:    Clouds form when warm air rises and cools. As the air rises, it expands and cools. If the air is humid, water vapor in the air condenses into liquid water droplets, forming clouds. This process is known
- control: Clouds form when warm air rises and cools. As the air rises, it expands and cools. If the air is humid, water vapor in the air condenses into liquid water droplets, forming clouds. This process is known

#### Turn 5
- strip:   A metal spoon feels colder than a wooden spoon because metal conducts heat away from your hand more efficiently than wood does.
- cold:    A metal spoon feels colder than a wooden spoon because metal conducts heat away from your hand more efficiently. Wood, on the other hand, is a better insulator, keeping your hand warmer.
- control: A metal spoon feels colder than a wooden spoon because metal conducts heat away from your hand more efficiently. Wood, on the other hand, is a better insulator, keeping your hand warmer.

#### Turn 6
- strip:   A rainbow appears after rain because the sunlight is refracted, reflected, and then refracted again as it passes through raindrops, dispersing the light into its component colors.
- cold:    A rainbow appears after rain because the sunlight is refracted, reflected, and then refracted again as it passes through raindrops, separating the white light into its component colors. This process creates a spectrum of
- control: A rainbow appears after rain because the sunlight is refracted, reflected, and then refracted again as it passes through raindrops. This causes the white light to separate into its component colors, creating a rainbow.

#### Turn 7
- strip:   Birds use a variety of methods to navigate long distances, including the sun, stars, the Earth's magnetic field, and even the Earth's gravitational pull. Some species also use landmarks and the Earth's magnetic field.
- cold:    Birds can find their way across long distances using a combination of celestial navigation, magnetic fields, and possibly even the Earth's magnetic field lines. They also use the position of the sun and stars to orient t
- control: Birds can find their way across long distances using a variety of methods, including the Earth's magnetic field, the position of the sun and stars, and even the sense of smell. Some species also use landmarks and the Ear

#### Turn 8
- strip:   Water boils at a lower temperature on a high mountain because the atmospheric pressure is lower at higher altitudes.
- cold:    Water boils at a lower temperature on a high mountain because the atmospheric pressure is lower. At higher altitudes, there's less air pressing down, so the boiling point of water is lower.
- control: Water boils at a lower temperature on a high mountain because the atmospheric pressure is lower. At higher altitudes, there's less air pressing down, so the boiling point of water is lower.

Early divergence from the cold run (before token 30): strip on 7 of 7 turns, control (no shift) on 4 of 7.

**Safe to enable NOVA_KV_STRIP_CONTEXT=1 on this machine: REVIEW** - the stripped reply broke within the first 3 tokens on turns [7] while the control did not; read the texts.
