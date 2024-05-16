## Financial Derivatives Project

This repository contains code for a financial derivatives project completed as part of a physics degree. The project involves analyzing the stock prices of Amazon (AMZN) and implementing various financial models to understand the stock's behavior, calculate investment performance, and estimate option prices.

### Project Structure

- **Data Analysis and Visualization:**
  - **Data:** The project starts by loading historical stock price data for Amazon (AMZN) from a CSV file.
  - **Histograms:** Histograms of daily returns are plotted to visualize the distribution of returns over different time periods (all data, last year, and last month).
  - **Normal Distribution Fit:** The best fit of the daily returns to a normal distribution is determined, and a histogram with the fitted normal distribution is plotted along with a QQ plot.
  - **Share Price with Major Events:** The project includes visualizations of Amazon's share price with major events such as acquisitions, earnings reports, and the COVID-19 pandemic.

- **Financial Analysis:**
  - **Drift and Volatility:** Annual drift and volatility of the share price are calculated.
  - **Investment Performance:** Investment performance is evaluated by calculating the final value, return on investment, and maximum/minimum values of the investment in Amazon stock. Additionally, the return on investment in a savings account is computed using the LIBOR rate.
  - **Investment Advice:** Based on volatility comparison, the code advises whether to invest in the stock or a savings account.

- **Binomial Model:**
  - **Option Pricing:** The binomial model is implemented to estimate option prices for call and put options on Amazon stock. The model calculates probabilities, up and down movements, and option prices for both European and American style options.

### Running the Code

1. **Requirements:**
   - Python 3.x
   - Libraries: pandas, numpy, matplotlib, scipy

2. **Data:**
   - Ensure you have historical stock price data for Amazon in a CSV file named 'AMZN.csv'. You can replace it with your own file if necessary.

3. **Execution:**
   - Run the Python script `AMZN_Final_Code.py`.
   - The script will generate various plots and print out analysis results.

### Sample Output

- Histograms of daily returns over different time periods.
- Histogram of daily returns with a fitted normal distribution.
- Share price evolution with major events marked.
- Option pricing for call and put options using the binomial model.

![AMZNTOTALSharePriceMAJOREVENTS](https://github.com/tgholmes/FinancialDerivatives-AMZNProject/assets/148396727/9ab67392-6d21-4c68-8a4b-ade7783b970f)
![AMZNWholeFoods](https://github.com/tgholmes/FinancialDerivatives-AMZNProject/assets/148396727/6a63dfce-30da-4617-8324-2f401de030b6)
![AMZNCOVID](https://github.com/tgholmes/FinancialDerivatives-AMZNProject/assets/148396727/d0f06092-88e9-4017-b53e-b105d33b7a1b)
![AMZNQuarterlyEarnings](https://github.com/tgholmes/FinancialDerivatives-AMZNProject/assets/148396727/3f239329-a0e8-42f1-8452-8f8f06f0cac7)
![AMZN_Histograms](https://github.com/tgholmes/FinancialDerivatives-AMZNProject/assets/148396727/061c8f62-b63c-44a3-b923-6624c2e60c24)
![PutOptionProfit](https://github.com/tgholmes/FinancialDerivatives-AMZNProject/assets/148396727/6cae54f1-557a-40cb-8ef7-6451ee6242de)
![CallOptionProfit](https://github.com/tgholmes/FinancialDerivatives-AMZNProject/assets/148396727/d8be3ed7-55b9-48ea-86f3-ddc233e17d16)

### Conclusion

This project offers insights into analyzing stock data and understanding financial derivatives using Python. It provides a foundation for further exploration and analysis of financial markets and instruments. Feel free to modify the code or extend it for your specific needs or research purposes.
