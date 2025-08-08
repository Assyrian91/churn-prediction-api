def evaluate_current_performance(self):
        """
        Evaluates the model's performance on the test data and returns the metrics.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1-score.
        """
        model, encoders, preprocessor = self.load_model_artifacts()

        test_data_path = self.config.get('paths.processed_test')
        test_data = pd.read_csv(test_data_path)

        target_col = self.config.get('data.target_column')
        X_test = test_data.drop(target_col, axis=1)
        y_test = test_data[target_col]

        categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen']
        
        for col in categorical_features:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype('object')

        # Use the preprocessor to transform the data
        X_test_transformed = preprocessor.transform(X_test)

        y_pred = model.predict(X_test_transformed)
        # Note: y_pred_proba is not used in the current version, so it can be removed if not needed.
        # y_pred_proba = model.predict_proba(X_test_transformed)

        # Calculate and store performance metrics
        performance_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, pos_label='Yes')),
            'recall': float(recall_score(y_test, y_pred, pos_label='Yes')),
            'f1_score': float(f1_score(y_test, y_pred, pos_label='Yes')),
            'evaluation_date': datetime.now().isoformat()
        }

        return performance_metrics