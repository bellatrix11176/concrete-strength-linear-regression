# Linear Regression — Concrete Compressive Strength (Train + Score)

This project builds a linear regression model to predict **concrete compressive strength** from mix ingredients and age, then applies the model to a separate scoring dataset to generate predictions.

It mirrors the RapidMiner workflow in Python, including:
- Setting `CompressiveStrength` as the label (target)
- Retaining `SlabID` as an identifier (not a predictor)
- Removing scoring observations that fall outside the training feature ranges
- Fitting a linear regression model
- Removing non-significant predictors (alpha = 0.05) and refitting
- Scoring the filtered scoring data set and exporting predictions

## Folder Structure

~~~text
Linear Regression/
|-- data/
|   |-- concrete_mix_strength_train.csv
|   |-- concrete_mix_strength_score.csv
|-- src/
|   |-- linear_regression_concrete_strength.py
|-- output/
|   |-- (generated files)
~~~

## How to Run

Open a terminal in the `Linear Regression/` folder (important for relative paths) and run:

~~~bash
python src/linear_regression_concrete_strength.py
~~~

## Outputs

After running the script, the following files are generated in `output/`:

- `concrete_strength_predictions.csv`
- `linear_regression_model_summary.txt`
- `final_model_pvalues.csv`
- `score_rows_removed_out_of_range.csv` (only if any scoring rows are removed)

MIT License

Copyright (c) 2026 Gina Aulabaugh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

🌐 **PixelKraze Analytics (Portfolio):** https://pixelkraze.com/?utm_source=github&utm_medium=readme&utm_campaign=portfolio&utm_content=homepage


