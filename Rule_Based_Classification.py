
############################################
# Classification for Calculating Potential Customer Return
############################################



###### The Business Problem #################

"""
- (level based) create new sales definitions and create new segments according to their definitions and create new
segments according to these segments. how much the company can earn on average from prospective customers wants to predict.

- For example: If you want to go to an All Inclusive hotel in Antalya during a busy period determine how much a customer
can earn on average is requested.

"""

######### The Data Set Story ###############

"""
You can see the prices of the sales made by the Gezinomi company and these contains information about sales. 
The data set consists of records generated in each sales transaction. is occurring.

SaleId: Sales ID
SaleDate: Date of the sale
Price: Price paid for the sale
ConceptName: Hotel concept information
SaleCityName: City where the hotel is located
CheckInDate: Customer's check-in date to the hotel
CInDay: Day of the week for the customer's hotel check-in
SaleCheckInDayDiff: Number of days between the check-in date and the sale date
Season: Season information for the check-in date at the hotel

"""

# import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.simplefilter(action="ignore")

# Adjusting Row Column Settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.2f' % x)

# Loading the Data Set
df = pd.read_excel("/kaggle/input/gzmn01/gezinomi.xlsx")

# Preliminary examination of the data set
def check_df(dataframe, head=5):
    print('##################### Shape #####################')
    print(dataframe.shape)
    print('##################### Types #####################')
    print(dataframe.dtypes)
    print('##################### Head #####################')
    print(dataframe.head(head))
    print('##################### Tail #####################')
    print(dataframe.tail(head))
    print('##################### NA #####################')
    print(dataframe.isnull().sum())
    print('##################### Quantiles #####################')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


# To drop the "SaleId" variable from the dataset
df = df.drop("SaleId", axis=1)

# To split the "SaleDate" variable into separate year, month, and day variables
df['SaleYear'] = df['SaleDate'].dt.year
df['SaleMonth'] = df['SaleDate'].dt.month
df['SaleDay'] = df['SaleDate'].dt.day

# To drop the "SaleDate" variable from the dataset
df = df.drop("SaleDate", axis=1)

# Filling the empty observations of the Price variable with the mean.
df['Price'].fillna(df['Price'].mean(), inplace=True)


# Examination of numerical and categorical variables
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numeric and categorical but cardinal variables in the data set.
    Note Categorical variables include categorical variables with numeric appearance.
    Parameters
    ------
         dataframe: dataframe
                Dataframe to get variable names
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables
    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                List of cardinal variables with categorical view
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 return lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == 'O']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)


# Categorical variable analysis
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('##########################################')
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


# Numerical variable analysis
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


# Correlation Analysis of the Variables
def df_corr(dataframe, annot=True):
    sns.heatmap(dataframe.corr(), annot=annot, linewidths=.2, cmap='Reds', square=True)
    plt.show(block=True)

def high_correlated_cols(dataframe,head=10):
    corr_matrix = dataframe.corr().abs()
    corr_cols = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                                   .astype(bool)).stack().sort_values(ascending=False)).head(head)
    return corr_cols

df_corr(df, annot=False)
high_correlated_cols(df, 15)

# Examination of the dependent variable
df["Price"].hist(bins=100)
plt.show(block=True)


# What is the total amount earned from sales by city?
df.groupby("SaleCityName").agg({"Price": "sum"})

# Calculating the total sales amount by grouping the data
sales_by_city = df.groupby("SaleCityName")["Price"].sum()

# Setting the x and y axes for the graph
x = sales_by_city.index
y = sales_by_city.values

# Setting the size of the graph
plt.figure(figsize=(10, 6))

# Plotting the bar chart
plt.bar(x, y)

# Adding axis labels and a title to the graph
plt.xlabel("Cities")
plt.ylabel("Total Sales Amount")
plt.title("Total Revenue by Cities")

# Rotating the axis labels
plt.xticks(rotation=45)

plt.ticklabel_format(style='plain', axis='y')

# Displaying the graph
plt.show(block=True)



# How much is earned by ConceptName types?
df.groupby("ConceptName").agg({"Price": "sum"})

# Calculating the average PRICE by grouping the data by SaleCityName and ConceptName
mean_price_by_city_concept = df.groupby(["SaleCityName", "ConceptName"]).agg({"Price": "mean"}).reset_index()

# Creating the plot
fig = px.bar(mean_price_by_city_concept, x="SaleCityName", y="Price", color="ConceptName",
             title="Average PRICE by City-Concept")

# Setting the axis labels
fig.update_layout(xaxis_title="City", yaxis_title="Average PRICE")

# Displaying the plot
fig.show(block=True)




# Şehir-Concept kırılımında PRICE ortalamaları nedir?
df.groupby(["SaleCityName", "ConceptName"]).agg({"Price": "mean"})

# satis_checkin_day_diff değişkenini EB_Score adında yeni bir kategorik değişken oluşturmada kullanılması
bins = [-1, 7, 30, 90, df["SaleCheckInDayDiff"].max()] # böünecek aralıklar belirlendi.
labels = ["Last Minutes", "Potential Planners", "Planners", "Early Bookers"] # aralıklara verilecek etiket isimlerinin belirlenmesi
df["EB_Score"] = pd.cut(df["SaleCheckInDayDiff"], bins, labels=labels) # yeni değişkenin oluşturulması
df.head()

#df.to_excel("eb_scorew.xlsx", index=False) # burada bu oluşturduğumuz yeni değişkenin excel dosyasına basılması



# Average salaries by City-Concept-EB Score
df.groupby(["SaleCityName", "ConceptName", "EB_Score"]).agg({"Price": ["mean", "count"]})

# Average salaries by City-Concept-EB Score
mean_count_price_by_city_concept_eb = df.groupby(["SaleCityName", "ConceptName", "EB_Score"]).agg({"Price": ["mean", "count"]}).reset_index()
mean_count_price_by_city_concept_eb.columns = ["SaleCityName", "ConceptName", "EB_Score", "Mean_Price", "Count"]

# Graph creation
fig = px.bar(mean_count_price_by_city_concept_eb, x="SaleCityName", y="Mean_Price", color="EB_Score",
             facet_col="ConceptName",
             title="Average Prices by City-Concept-EB Score",
             labels={"EB_Score": "EB Score", "Mean_Price": "Average Price"},
             hover_data=["Count"])

# Updating axis labels
fig.update_layout(xaxis_title="SaleCityName", yaxis_title="Average Price")

# Showing the graph
fig.show(block=True)



# Average prices by City-Concept-Season breakdown
df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": ["mean", "count"]})

# Calculating the mean prices and counts by SaleCityName, ConceptName, and Seasons
mean_count_price_by_city_concept_seasons = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": ["mean", "count"]}).reset_index()
mean_count_price_by_city_concept_seasons.columns = ["SaleCityName", "ConceptName", "Seasons", "Mean_Price", "Count"]

# Creating the plot
fig = px.bar(mean_count_price_by_city_concept_seasons, x="SaleCityName", y="Mean_Price", color="Seasons",
             facet_col="ConceptName",
             title="Average Prices by City-Concept-Season",
             labels={"Seasons": "Season", "Mean_Price": "Average Price"},
             hover_data=["Count"])

# Updating axis labels
fig.update_layout(xaxis_title="SaleCityName", yaxis_title="Average Price")

# Showing the plot
fig.show(block=True)





#Average Prices by City-Concept-CInDay Breakdown
df.groupby(["SaleCityName", "ConceptName", "CInDay"]).agg({"Price": ["mean", "count"]})

# Calculating the mean prices and counts by SaleCityName, ConceptName, and CInDay
mean_count_price_by_city_concept_cinday = df.groupby(["SaleCityName", "ConceptName", "CInDay"]).agg({"Price": ["mean", "count"]}).reset_index()
mean_count_price_by_city_concept_cinday.columns = ["SaleCityName", "ConceptName", "CInDay", "Mean_Price", "Count"]

# Creating the plot
fig = px.bar(mean_count_price_by_city_concept_cinday, x="SaleCityName", y="Mean_Price", color="CInDay",
             facet_col="ConceptName",
             title="Average Prices by City-Concept-CInDay",
             labels={"CInDay": "CInDay", "Mean_Price": "Average Price"},
             hover_data=["Count"])

# Updating axis labels
fig.update_layout(xaxis_title="SaleCityName", yaxis_title="Average Price")

# Showing the plot
fig.show(block=True)


# Average Price by Year, Concept, and City
df.groupby(["SaleYear", "ConceptName", "SaleCityName"])["Price"].mean().reset_index()

# Grouping and calculating mean prices
mean_price_by_year_concept_city = df.groupby(["SaleYear", "ConceptName", "SaleCityName"])["Price"].mean().reset_index()

# Creating the chart
fig = px.bar(mean_price_by_year_concept_city, x="SaleYear", y="Price", color="ConceptName",
             facet_col="SaleCityName", title="Average Price by Year, Concept, and City",
             labels={"SaleYear": "Year", "Price": "Average Price"})

# Updating axis labels
fig.update_layout(xaxis_title="Year", yaxis_title="Average Price")

# Displaying the chart
fig.show(block=True)

# Grouping and calculating mean prices
mean_price_by_city_concept_seasons = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False).reset_index()

# Creating the chart
fig = px.bar(mean_price_by_city_concept_seasons, x="SaleCityName", y="Price", color="ConceptName",
             facet_col="Seasons", title="Average Price by City, Concept, and Seasons",
             labels={"SaleCityName": "City", "Price": "Average Price", "ConceptName": "Concept", "Seasons": "Season"})

# Updating axis labels
fig.update_layout(xaxis_title="City", yaxis_title="Average Price")

# Displaying the chart
fig.show(block=True)



# Sorting the output of the City-Concept-Season breakdown by PRICE
df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False).head(20)
agg_df = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False).head(20)

agg_df.reset_index(inplace=True)

agg_df.head()


# Defining new level based sales and adding them as variables to the data set
agg_df["sales_level_based"] = agg_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '_'.join(x).upper(), axis=1)

agg_df.head()

# Segmentation of persons
# Segmentation by price
agg_df["SEGMENT"] = pd.cut(agg_df["Price"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)

# Calculation of average max, min and total values in segment breakdown
agg_df.groupby("SEGMENT").agg({"Price": ["mean", "max", "min", "sum"]})

# Sorting agg_df by price variable
agg_df.sort_values(by="Price")

# example
new_user = "ANTALYA_HERŞEY DAHIL_HIGH"
agg_df[agg_df["sales_level_based"] == new_user]