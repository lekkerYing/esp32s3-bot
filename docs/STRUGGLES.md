# ESP32 bot: what we tried, what went wrong, what we changed

Written from our change log (22 Oct to 18 Nov 2026). Numbers are for the test question "Who is Fei-Fei Li?" unless it says otherwise.

## 1. Speed: from 26 minutes to under 7

| Step | What we tried | Result |
| --- | --- | --- |
| Start | 135M model, one word at a time | 26 min for 30 words |
| Guess ahead | Check 15 guessed words in one pass over the model (speculative decoding) | 7 min 22 s |
| Faster maths | Chip's own dot product routine (esp-dsp), input kept in internal RAM | 4 min 25 s |
| Read ahead | One core reads the next layer from the SD while the other calculates | 2 min 55 s |
| Bigger model | Ambika's smolFem 360M | first try about 2 min per word (see below) |
| Own answers as guesses | Rebuilt the guess table from the 4-bit model's own answers | 7 min 37 s |
| Merged question pass, batch of 32 | The question pass also checks the first guesses | 6 min 41 s |

Things that went wrong on the way:
- **Plain 4-bit compression ruined the 360M model.** Word accuracy on the dataset fell from 71.8% to about 50%. Fix: retrain the 4-bit copy to copy the full model (quantization-aware distillation): 71.0%.
- **The guess table did not match the 4-bit model.** It held the dataset answers, but the 4-bit model words things a bit differently, so the guesses missed. Fix: build the table from the 4-bit model's own answers.
- **Lowercase or slightly different questions are very slow.** "who is Fei-Fei Li?" does not match the guess table, so the board writes one word per pass (about 2 min per word). Tip: start with a capital letter and type the question exactly.
- **Questions that are not in the dataset take about an hour.** Every word needs the whole 224 MB model read from the SD card. An off-dataset question took 2 h 19 min for 73 words and rambled. This is why answers are capped at 30 words.
- **The SD card is the limit.** We measured 1.5 MB/s, a bit under the 1.8 MB/s we first measured. Raising the SPI clock from 20 to 40 MHz gave exactly the same speed (`/sdspeed` test). Not tried: another card, bigger clusters, 4-bit SD mode.

## 2. Power: brownouts and resets

- **Brownouts under load** (`BROWNOUT_RST`, sometimes in a loop at boot). Tried: new USB cable, dimmed backlight, lower Bluetooth power. What actually helped: a second USB-C port with a phone charger. With the laptop port alone the board browned out at 6.5 min; with both ports it ran 90 minutes without a reset. The laptop port was too weak.
- **Bluetooth plus a big batch tipped the supply over.** Resets only happened with Bluetooth on and a batch of 24 or more. Fix: lower Bluetooth transmit power and one advertisement per second instead of about 10.
- **PiSugar 3 through a homemade USB-C cable failed.** Only the backlight came on. Replacing the cable with direct wires to the 5V pin fixed it.
- **16 restarts during the 8-hour run, 13 from the task watchdog.** Cause not proven. We raised the watchdog timeout from 5 s to 20 s and added position codes in RTC memory that are written to `/boots.csv`, so we can see where it was stuck. Thanks to `/batch_pos.txt` and `/autorun.txt` the list still finished after every restart.

## 3. Bluetooth

- **The phone lost the connection partway through an answer** (three times). The board kept working on its screen. We added connect/disconnect logging and a longer supervision timeout (6 s); after that a connection held for 9+ minutes. Not proven that this was the fix (one run).
- **The phone dropped again after about 40 minutes** on very long answers, with no reason code from the NimBLE stack. Not solved. Workaround: read the answer on the screen, serial, or the SD log.
- Part of "it stops half way" was simply the 30-word limit on a question without `/full`.

## 4. Hardware

- **The common 5 V SD card reader did not work.** On 3V3 the card answered but never started, and the 5V0 pin does not give out the USB 5 V. Only a 3.3 V reader worked. This is why the order list has a 3.3 V reader.
- **The pin poster from the board's seller was off by one row.** Read the small numbers printed next to each pin hole on the board instead of the coloured boxes on the poster.
- **The screen went black or the SD mount failed several times.** Every time it was a loose or broken wire or solder joint on a shared 3V3 or GND line, not the code. Fix: solder everything on a protoboard with all 3V3 joined and all GND joined.
- **Screen pins changed three times.** Original DIN 40 / CLK 39 / DC 42 / RST 2; then a swap of CLK and DIN; then on 9 Nov the layout we use now (DIN 2, CLK 42, CS 41, DC 40, RST 39). The backlight wire moved from GPIO 14 (dimmed) to 3V3 (full brightness), which raised the power draw. The old 497 mW idle value is no longer valid.
- **The screen only works with the USB-C cable between the ESP32 and the PiSugar.** We don't know why. The cable may bypass the INA219, so idle should be measured again with `/idle`.
- **The INA219 was not found at first,** then it was wired and detected. One test run logged 0 V and 0 mA, so it was not in the power line that time.

## 5. Measuring energy

- **The board has no clock.** The SD log has epoch 0 until you send `/time`. For the comparison the 10 runs were placed one after the other from 13 Nov 18:00 (approximate).
- **The Pi and the ESP32 are not measured at the same point.** The Pi's power data is read at the PiSugar input (solar side), the ESP32's INA219 sits at the PiSugar output. Compare the pattern, not single numbers.
- **The ESP32 idle draw is not measured again** with the current wiring.
- **Two test runs left no rows on the SD card.** Windows may have shown a stale card or the runs did not finish. Cause not found.
