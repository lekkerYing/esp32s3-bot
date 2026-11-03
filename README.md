# SmolFem 360M on an ESP32-S3 N16R8

Runs a 4-bit version of smolFem (a LoRA fine-tune of SmolLM2-360M-Instruct) on an ESP32-S3 DevKitC-1 board with the WROOM-1 N16R8 module. The model is streamed layer by layer from an SD card. The full guide is in [docs/HOWTO.md](docs/HOWTO.md): wiring, tools, flashing, asking questions and measuring energy.

    sketch/smol_360/   Arduino sketch, partitions.csv, draft_data.h (guess table)
    scripts/           qat360.py, export_sd360.py, make_draft360.py
    docs/HOWTO.md      the guide

Run the scripts from the `scripts` folder, in this order. They read settings from environment variables: `SMOLFEM_DIR` (the full smolFem model folder), `DATA_DIR` (the dataset folder with the cleaned train and valid files), `OUT_DIR` (where the exported files go).

1. `python qat360.py` shrinks the model to 4 bit
2. `python export_sd360.py` writes `model.bin`, `bpe.bin` and `head.bin`
3. `python make_draft360.py` rebuilds `draft_data.h`

`model.bin`, `bpe.bin` and `head.bin` are not stored in git (`*.bin` is ignored). On the SD card, `model.bin` and `bpe.bin` go in a folder called `SMOL360`. `head.bin` is written to the board's flash at 0x210000.

The scripts need Python with torch, transformers and numpy. They were written on Windows and tested there.
