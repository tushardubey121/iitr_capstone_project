## Data Findings
- Express deliveries have the highest delay rate (48.97%), followed by Standard (20.56%) and Economy (8.57%).
- Delays increase with hub tier and delivery distance. Parcels travelling over 601 km have an 85.23% delay rate.
- The strongest factors associated with delivery time are promised_hours (0.8372), distance_km (0.4013) and n_stops (0.3247).
- Log transformation reduced the skewness of declared parcel value, making its mean and median nearly equal.

## Regression Results
- The baseline model performed poorly (R²: -0.0212).
- Linear Regression achieved the best validation RMSE of 4.8912 hours.
- Decision Tree performed best at depth 6, while unlimited depth caused overfitting.
- Gradient Boosting achieved the best cross-validation R² of 0.9353.
- Recommendation: Use Gradient Boosting for production because of its stronger cross-validation performance.

## Classification Results
- The baseline classifier achieved 76.67% accuracy but failed to detect any delayed parcels.
- KNN performed best at K=7, with an F1 score of 0.8333.
- Hyperparameter tuning did not improve performance.
- A 0.5 threshold was selected to balance precision (79.55%) and recall (83.33%) on validation.
- Test performance: 68.89% precision, 75.61% recall and 0.7209 F1.
- Recommendation: Retain the untuned Random Forest with a 0.5 threshold and monitor its performance on new data.

## Operations Recommendations
- Prioritise high-delay hubs, especially Guwahati (62%), Agartala (54%), Raipur (48%), Nagpur and Patna (46%).
- Review hub tiers where actual performance differs from assigned classifications.
- Of 48 new parcels, 8 were flagged, including 6 from Escalate hubs. These require hub manager attention.
- K=3 provides the best practical segmentation for the three existing operational playbooks.

## Production Plan
- Model predictions and evaluation metrics are saved in the outputs folder.
- Model reload testing confirms identical predictions across 180 rows.
- Dependencies and random states are fixed for reproducibility.
- Shared preprocessing and feature engineering are centralised in src/shipment_prep.py.
- New parcel predictions and recommended actions are saved in outputs/new_shipment_actions.csv.

## Monitoring and Retraining
- Investigate weekly delay rates outside 18–30% for two consecutive normal weeks.
- Investigate flagged parcel changes above 10 percentage points.
- Retrain if four-week recall falls below 0.70.
- Retrain monthly using the latest delivered parcel data.
- Compare sale-period performance with previous sales to account for seasonal effects.

## Limitations
- Synthetic data may not reflect real-world delivery challenges.
- The test set contains only 41 delayed parcels, making performance metrics sensitive to individual predictions.
- Model selection and threshold decisions are based on a single validation split.

## Final Recommendation
Use Gradient Boosting for delivery-time prediction and Random Forest with a 0.5 threshold for delay detection. Combine both with hub-level monitoring and regular retraining to identify delays early and improve operational decisions.