import pandas as pd
import json
#from evidently.utils.data_definition import ColumnMapping
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import os
from datetime import datetime
from config_manager import ConfigManager

class DataMonitor:
    def __init__(self):
        self.config = ConfigManager()
        
    def generate_data_drift_report(self):
        """ Data Drift"""
        
        reference_data = pd.read_csv(self.config.get('monitoring.reference_data_path'))
        current_data = pd.read_csv(self.config.get('monitoring.current_data_path'))
        
        # Column Mapping
        target_col = self.config.get('data.target_column')
        column_mapping = ColumnMapping()
        column_mapping.target = target_col
        
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        data_drift_report.run(reference_data=reference_data, 
                            current_data=current_data,
                            column_mapping=column_mapping)
        
        reports_dir = self.config.get('monitoring.reports_path')
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{reports_dir}/data_drift_report_{timestamp}.html"
        
        data_drift_report.save_html(report_path)
        
        print(f"Data drift report saved: {report_path}")
        return report_path
    
    def check_data_drift(self):
        reference_data = pd.read_csv(self.config.get('monitoring.reference_data_path'))
        current_data = pd.read_csv(self.config.get('monitoring.current_data_path'))
        
        drift_detected = False
        drift_summary = {}
        
        for col in reference_data.select_dtypes(include=['number']).columns:
            if col in current_data.columns:
                ref_mean = reference_data[col].mean()
                curr_mean = current_data[col].mean()
                
                if abs(ref_mean - curr_mean) / ref_mean > 0.1:
                    drift_detected = True
                    drift_summary[col] = {
                        'reference_mean': ref_mean,
                        'current_mean': curr_mean,
                        'drift_percentage': abs(ref_mean - curr_mean) / ref_mean * 100
                    }
        
        return {
            'drift_detected': drift_detected,
            'drift_summary': drift_summary,
            'check_date': datetime.now().isoformat()
        }

if __name__ == "__main__":
    monitor = DataMonitor()
    
    report_path = monitor.generate_data_drift_report()
    
    drift_check = monitor.check_data_drift()
    print("Drift Check Results:", json.dumps(drift_check, indent=2))
