# Vector Database Directory

This folder is used to store the generated vector database files (e.g., `.index`, `.pkl`, and `.json` files) created by N.O.V.A.'s ingestion engine.

**Why is this folder empty in the repo?**
The database files are excluded from version control because they can become very large, cause merge conflicts, and are essentially compiled artifacts. 

**How to generate the database:**
You do not need to download anything for this folder. N.O.V.A. will automatically recreate the database files here the first time you run the backend, as long as your source documents are present in your `scan_root` folder.