# NOTER

## Corr

The provided CSV file contains a correlation matrix for various variables related to distances and times associated with different locations such as doctor, dentist, pharmacy, kindergarten/nursery, school, supermarket, library, train station, and sports facility. Here's a summary of the main findings and patterns:

1. **High Correlation within Same Location Variables**:
   - Variables related to the same location tend to have high correlations with each other. For example, `walkDistance_doctor_1`, `duration_min_doctor_1`, and `total_time_min_doctor_1` all have high correlations with each other.
   - This pattern is consistent across all locations, indicating that walking distance, absolute distance, duration, and total time are closely related for each specific location.

2. **Walking Distance and Duration**:
   - Walking distance and duration are highly correlated for each location. This is expected as longer distances generally take more time to traverse.
   - For example, `walkDistance_doctor_1` and `duration_min_doctor_1` have a correlation of 0.98.

3. **Total Time and Duration**:
   - Total time and duration for each location are also highly correlated. This suggests that the duration of travel is a significant component of the total time spent.
   - For example, `duration_min_doctor_1` and `total_time_min_doctor_1` have a correlation of 0.95.

4. **Waiting Time**:
   - Waiting time variables generally have lower correlations with other variables, indicating that waiting times are more independent of walking distances and durations.
   - For example, `wait_time_dest_min_doctor_1` has a relatively low correlation with `walkDistance_doctor_1` (-0.11).

5. **Cross-Location Correlations**:
   - There are moderate correlations between similar variables across different locations. For example, `walkDistance_doctor_1` and `walkDistance_dentist_1` have a correlation of 0.62.
   - This suggests that locations that are closer to one type of facility (e.g., doctor) might also be closer to other types of facilities (e.g., dentist).

6. **Total Waiting Time and Total Travel Time**:
   - `total_waiting_time` has moderate correlations with waiting times at individual locations but lower correlations with walking distances and durations.
   - `total_travel_time` has high correlations with most duration and distance variables across all locations, indicating it is a comprehensive measure of travel effort.

7. **Total Time**:
   - `total_time` is highly correlated with `total_travel_time` (0.98) and also shows high correlations with most individual location variables, reflecting its role as an aggregate measure of time spent.

In summary, the main patterns indicate strong relationships within variables of the same location and moderate relationships across different locations. Waiting times are generally less correlated with distance and duration variables, highlighting their independent nature. Total travel time and total time serve as comprehensive measures that correlate highly with most individual variables.



***

- Wait time at different destinations have fairly low correlations
- Distances have fairly high correlations
- Same for distances

***

## FOR HEX AGGREGATED DATA

The correlation matrix you provided shows the relationships between various distance and time metrics for different types of locations such as doctors, dentists, pharmacies, schools, and others. Here's a summary of the patterns observed:

1. **Strong Positive Correlations:**
   - **Walking Distance, Absolute Distance, and Duration:** These metrics are generally strongly positively correlated across different types of locations. For example, `walkDistance_doctor_1` has high correlations with `duration_min_doctor_1` (0.95) and `total_time_min_doctor_1` (0.87). This pattern is consistent across other location types like dentists, pharmacies, and schools.
   - **Total Time Metrics:** The total time metrics for different locations are also highly correlated with each other. For example, `total_time_min_doctor_1` is highly correlated with `total_time_min_dentist_1` (0.62) and `total_time_min_pharmacy_1` (0.59).

2. **Moderate Positive Correlations:**
   - **Inter-location Correlations:** There are moderate positive correlations between similar metrics across different locations. For example, `walkDistance_doctor_1` and `walkDistance_dentist_1` have a correlation of 0.63.
   - **Duration and Total Time:** Duration and total time for the same location type are moderately to strongly correlated, indicating that longer durations generally lead to longer total times.

3. **Weak or Negative Correlations:**
   - **Waiting Time:** Waiting times at destinations (`wait_time_dest_min`) generally show weak or negative correlations with other metrics. For example, `wait_time_dest_min_doctor_1` has a weak correlation with `walkDistance_doctor_1` (-0.2) and `duration_min_doctor_1` (-0.05).
   - **Total Waiting Time:** The total waiting time has weak correlations with most other metrics, indicating that waiting times are somewhat independent of travel distances and durations.

4. **Specific Observations:**
   - **Pharmacy Metrics:** Pharmacy-related metrics like `walkDistance_pharmacy_1` and `duration_min_pharmacy_1` show strong correlations (0.91), indicating that walking distance is a significant factor in the duration to reach a pharmacy.
   - **Kindergarten/Nursery Metrics:** Metrics related to kindergartens and nurseries also show strong internal correlations, such as `walkDistance_kindergarten_nursery_1` and `duration_min_kindergarten_nursery_1` (0.98).

5. **Overall Trends:**
   - **Consistency Across Locations:** The patterns of strong correlations between walking distance, absolute distance, and duration are consistent across different types of locations.
   - **Total Travel and Total Time:** The total travel time and total time metrics are highly correlated (0.98), indicating that the majority of the total time is composed of travel time.

In summary, the correlation matrix reveals that walking distances, absolute distances, and durations are strongly interrelated across various locations. Waiting times tend to be less correlated with these metrics, suggesting they are influenced by different factors.