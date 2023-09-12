# Advisor-Valuation-Run-Guide
This is an outline of using this python code to quickly do some data manipulation for Advisor Valuations. Do the following steps to be able to run the code
#### 1. Install the following python packages if you dont have them already:
  - pandas
  - datetime
  - fuzzywuzzy <br><br>
You can install these packages by doing py -m pip install "said package". You will also need to have pip installed to be able to use the installtion method. "py" should be
installed if you downloaded python onto your computer already. Make sure when you run the install, you are at the base directory so that you don't need to re-install any packages.
#### 2. Once you have installed all the packages, change the following values to meet your needs:
  - advisor_name = "Change the name here"
  - client_summary_df = pd.read_csv("change file directory here")
  - revenue_by_client_df = pd.read_csv("change file directory here")
  - Percent_Non_Recurring (defaulted to 0.99 aka 99%. Can be found on line 130)
  - Percent_Reccuring_Rev (defaulted to 0.01 aka 1%. Can be found on line 128)
  - Anything else that you feel needs modification in the code (such as Reccuring_Rev_Multiple and Non_Recurring_Multiple found on lines 127 and 129).
<br> <br>  *Note* - wherever the file is located on your computer, thats what will need to go in the "change file directory here". You may also need to change the "read_csv" to "read_excel"
if the file(s) you are trying to manipulate are of ".xlsx" or ".xls".
#### 3. When you run the code, there will be a couple of questions asked of you in the terminal to fill out to make sure the valuation is fully correct
  1. Advisor ownership percentage. Make sure to put just the number from 0-100
  2. Risk Adjustment: Proactive Service Model (yes/no)
  3. Risk Adjustment: Reactive Service Model (yes/no)
  4. Risk Adjustment: <5% YOY Growth last 3 years (yes/no)
#### 4. (optional) The output is going to be of type csv. If you wish to change that, you will need to change ".to_csv" to ".to_excel" to output an excel file.

