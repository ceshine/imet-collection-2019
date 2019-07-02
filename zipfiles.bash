rm imet.7z
7za a -bd -mx=0 imet.7z imet/*.py setup.py
rm helperbot.7z
cd pytorch_helper_bot
7za a -bd -mx=0 ../helperbot.7z helperbot/*.py *.py
