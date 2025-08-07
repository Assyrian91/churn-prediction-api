import pandas as pd
import pickle
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import os
from config_manager import ConfigManager
import mlflow
from mlflow.tracking import MlflowClient

class ModelPerformanceMonitor:
    def __init__(self):
        self.config = ConfigManager()
        self.client = MlflowClient()
        
    def load_model_artifacts(self):
        """تحميل النموذج والـ artifacts"""
        models_dir = self.config.get('paths.models_dir')
        
        with open(f'{models_dir}/churn_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open(f'{models_dir}/encoder.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        with open(f'{models_dir}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        return model, encoders, scaler
    
    def evaluate_current_performance(self):
        """تقييم أداء النموذج الحالي"""
        
        # تحميل النموذج
        model, encoders, scaler = self.load_model_artifacts()
        
        # تحميل بيانات الاختبار
        test_data = pd.read_csv(self.config.get('paths.processed_test'))
        
        # تحضير البيانات
        target_col = self.config.get('data.target_column')
        X_test = test_data.drop(target_col, axis=1)
        y_test = test_data[target_col]
        
        # معالجة البيانات
        for col, encoder in encoders.items():
            if col in X_test.columns:
                X_test[col] = encoder.transform(X_test[col])
        
        X_test_scaled = scaler.transform(X_test)
        
        # التنبؤ
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # حساب Metrics
        performance_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, pos_label='Yes')),
            'recall': float(recall_score(y_test, y_pred, pos_label='Yes')),
            'f1_score': float(f1_score(y_test, y_pred, pos_label='Yes')),
            'evaluation_date': datetime.now().isoformat()
        }
        
        return performance_metrics
    
    def track_performance_history(self):
        """تتبع تاريخ الأداء"""
        metrics_file = "metrics/performance_history.json"
        
        # الحصول على الأداء الحالي
        current_metrics = self.evaluate_current_performance()
        
        # تحميل التاريخ السابق
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # إضافة الأداء الحالي
        history.append(current_metrics)
        
        # الاحتفاظ بآخر 30 تقييم فقط
        history = history[-30:]
        
        # حفظ التاريخ
        os.makedirs('metrics', exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def check_performance_degradation(self, threshold=0.05):
        """فحص تدهور الأداء"""
        history = self.track_performance_history()
        
        if len(history) < 2:
            return False, "Not enough history to compare"
        
        # مقارنة آخر تقييم مع المتوسط السابق
        current_f1 = history[-1]['f1_score']
        previous_f1_scores = [h['f1_score'] for h in history[:-1]]
        avg_previous_f1 = np.mean(previous_f1_scores)
        
        degradation = avg_previous_f1 - current_f1
        
        if degradation > threshold:
            return True, f"Performance degraded by {degradation:.4f} (threshold: {threshold})"
        
        return False, f"Performance stable. Degradation: {degradation:.4f}"
    
    def generate_performance_report(self):
        """إنشاء تقرير الأداء"""
        current_metrics = self.evaluate_current_performance()
        degradation_check = self.check_performance_degradation()
        history = self.track_performance_history()
        
        report = {
            'current_performance': current_metrics,
            'performance_degradation': {
                'detected': degradation_check[0],
                'details': degradation_check[1]
            },
            'history_summary': {
                'total_evaluations': len(history),
                'avg_accuracy': np.mean([h['accuracy'] for h in history]),
                'avg_f1_score': np.mean([h['f1_score'] for h in history])
            }
        }
        
        # حفظ التقرير
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/performance_report_{timestamp}.json"
        
        os.makedirs('reports', exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved: {report_path}")
        return report

if __name__ == "__main__":
    monitor = ModelPerformanceMonitor()
    report = monitor.generate_performance_report()
    print("Performance Report:", json.dumps(report, indent=2))
