# Results: 10 questions on the ESP32

Files copied from the SD card after the batch run of 13 to 14 November 2026 (30 word limit, powered by a PiSugar 3 through an INA219).

    questions.txt         the 10 questions
    energy_questions.csv  one row per answered question: words, seconds, mWh, average mW, peak mA, volts
    energy_answers.csv    the answers
    energy_samples.csv    one INA219 reading every 5 s
    energy_idle.csv       one idle measurement (older, with a dimmed backlight, so not comparable)
    boots.csv             why the board restarted (power on, watchdog, brownout)

Notes:
* The board has no clock, so rows are told apart by boot id, not by date.
* `energy_questions.csv` also holds a few earlier test rows. The 10 runs of the comparison are the boots 5013 to 9589 (see the guide for the totals: 7.8 hours and about 7.8 Wh).
* The board restarted 16 times during the run, 13 times from the task watchdog and 3 times from a brownout. The batch carried on every time.
