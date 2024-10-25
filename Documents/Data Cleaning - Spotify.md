# **Data Cleaning Report for Spotify Dataset**

This report outlines the data cleaning steps performed on the Spotify dataset to ensure accuracy, completeness, and usability for subsequent analysis. Each section breaks down the methods used to clean and transform the data, organized by key data cleaning tasks.

---

## **1\. Data Loading and Initial Exploration**

The dataset was loaded in Parquet format and initially explored to gain an understanding of its structure and to identify potential issues.

`import pandas as pd`  
`df = pd.read_parquet('0000 (1).parquet')`  
`print(df.head())`  
`df.shape`

* **Shape**: The dataset contains **114,000 rows** and **21 columns**.  
* **Columns**: Initial inspection of `head()` revealed basic information, including column names, sample values, and column types, which were further investigated in the following steps.

---

## **2\. Handling Null Values**

### **2.1 Identifying Rows with Null Values in Key Columns**

To ensure the dataset contained meaningful information, the focus was placed on key columns, particularly `artists`, which plays a critical role in analyzing track metadata.

`null_artists = df[df['artists'].isnull()]`  
`print(null_artists)`

* **Observation**: Rows with null values in `artists` were identified. These rows were found to have null values in all three essential columns: `artists`, `album_name`, and `track_name`.

### **2.2 Dropping Null Values in Essential Columns**

`df.dropna(subset=['artists'], inplace=True)`

* **Action**: All rows with missing `artists` values were dropped in place, ensuring that only complete records were retained for analysis. This decision was based on the fact that null values in `artists` reduce the value of the row for identifying track information.  
* **Verification**: `df.isnull().sum()` confirmed that there were no remaining null values in the dataset.

---

## **3\. Duplicate Detection and Removal**

### **3.1 Identifying Duplicate Rows**

Duplicate rows were detected based on the `track_id` column to ensure unique entries for each track.

`duplicates = df[df.duplicated(subset=['track_id'], keep=False)]`  
`print(duplicates)`

* **Observation**: **40,900 duplicate rows** were identified based on `track_id`, meaning several tracks appeared multiple times in the dataset.

### **3.2 Sorting and Reviewing Duplicate Rows**

To examine duplicate records more easily, they were sorted by `track_id` so that identical entries appeared next to each other.

`sorted_duplicates = duplicates.sort_values(by='track_id')`  
`print(sorted_duplicates)`

* **Action**: By reviewing the sorted duplicates, it was confirmed that keeping the first occurrence of each unique `track_id` would be the most appropriate way to handle duplicates.

### 

### **3.3 Removing Duplicate Rows**

`df_cleaned = df.drop_duplicates(subset=['track_id'], keep='first')`  
`print(df_cleaned)`

* **Action**: Duplicate entries were removed, retaining only the first instance of each unique `track_id`.  
* **Result**: The cleaned dataset now contains **89,740 rows**, down from 114,000, reflecting the removal of duplicate tracks.

---

## **4\. Data Type and Unique Value Validation**

### **4.1 Verifying Data Types**

`df_cleaned.info()`

* **Observation**: This step provided an overview of data types and non-null counts across all columns. Initial inspection indicated that data types were appropriate for each column; for instance, `popularity` and `duration_ms` were integers, while features like `danceability` and `valence` were floats.

### **4.2 Checking for Unique Values**

The number of unique entries in each column was calculated to understand variability, particularly in categorical columns.

`unique_counts = df_cleaned.nunique()`  
`print(unique_counts)`  
`print(df_cleaned['track_genre'].unique())`

* **Action**: This analysis confirmed the number of distinct values, particularly focusing on `track_genre` and `artists` to understand the dataset's diversity. A unique set of genres was printed to ensure data integrity for genre-based analysis.

---

## **5\. Saving the Cleaned Data**

The cleaned data was saved to a new CSV file for easy access in subsequent stages.

`df_cleaned.to_csv('df_cleaned.csv', index=False)`  
`df_cleaned.shape`

* **Action**: The cleaned dataset was saved as `df_cleaned.csv`, with a final shape of **89,740 rows** and **21 columns**.  
* **Outcome**: The data is now in a ready-to-use format for further exploration and analysis.

---

## **Summary of Data Cleaning Steps**

1. **Null Value Handling**: Rows with null values in the essential `artists` column were dropped.  
2. **Duplicate Removal**: Removed duplicate entries based on the unique `track_id`.  
3. **Data Type Validation**: Ensured appropriate data types across columns.  
4. **Unique Value Verification**: Assessed unique values to validate data consistency, especially in categorical fields.  
5. **Export of Clean Data**: Saved the final cleaned dataset to `df_cleaned.csv` for further analysis.

Each of these steps contributed to a cleaner, more reliable dataset, ensuring a solid foundation for accurate and insightful analysis.

