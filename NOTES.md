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