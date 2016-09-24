function process_dir_fn {
     ../convert-rc.sh questions/validation validation.txt
     ../convert-rc.sh questions/training training.txt
     ../convert-rc.sh questions/test test.txt
}

echo "Processing CNN dataset"
cd cnn
process_dir_fn

echo "Processing DailyMail dataset"
cd ../dailymail
process_dir_fn
