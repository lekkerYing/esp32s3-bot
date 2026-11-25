# SmolFem 360M on an ESP32-S3 N16R8

Runs a 4-bit version of smolFem (a LoRA fine-tune of SmolLM2-360M-Instruct on Ambika's feminist dataset) on an ESP32-S3 DevKitC-1 board with the WROOM-1 N16R8 module. The model is streamed layer by layer from an SD card. The full guide is in [docs/HOWTO.md](docs/HOWTO.md): wiring, tools, flashing, asking questions and measuring energy.

    sketch/smol_360/   Arduino sketch, partitions.csv, draft_data.h (guess table)
    scripts/           qat360.py, export_sd360.py, make_draft360.py
    docs/HOWTO.md      the guide
    docs/STRUGGLES.md  what we tried, what went wrong and what we changed
    results/           the SD card files of the 10 question run

Run the scripts from the `scripts` folder, in this order. They read settings from environment variables: `SMOLFEM_DIR` (the full smolFem model folder), `DATA_DIR` (the dataset folder with the cleaned train and valid files), `OUT_DIR` (where the exported files go).

1. `python qat360.py` shrinks the model to 4 bit
2. `python export_sd360.py` writes `model.bin`, `bpe.bin` and `head.bin`
3. `python make_draft360.py` rebuilds `draft_data.h`

`model.bin`, `bpe.bin` and `head.bin` are not stored in git (`*.bin` is ignored). Download them from Hugging Face: https://huggingface.co/LekkerYing/smolfem-360m-esp32. On the SD card, `model.bin` and `bpe.bin` go in a folder called `SMOL360`. `head.bin` is written to the board's flash at 0x210000.

The scripts need Python with torch, transformers and numpy. They were written on Windows and tested there.

The results are written up as a post on the Sun Shines Bright site (comparison with the solarbot on a Raspberry Pi 5): https://github.com/computational-mama/sunshinesbright/tree/main/content/posts/esp32-comparison

Wiring of the build: [docs/images/wiring-esp32.png](docs/images/wiring-esp32.png)

Licence: MIT, see [LICENSE](LICENSE).
