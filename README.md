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

**Case Studies:** https://pixelkraze.com/case-studies?utm_source=github&utm_medium=readme&utm_campaign=portfolio&utm_content=case_studies_link
**Colab Notebook:** https://YOUR-COLAB-LINK?utm_source=github&utm_medium=readme&utm_campaign=novawireless&utm_content=colab_button

