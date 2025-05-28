import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ExplanationVisualizer:
    """Visualizes SHAP explanations for adherence predictions."""
    
    def __init__(self):
        self.colors = {
            "wearable": "#2ecc71",  # Green
            "ehr": "#3498db",       # Blue
            "survey": "#e74c3c",    # Red
            "temporal": "#f1c40f",  # Yellow
            "behavioral": "#9b59b6"  # Purple
        }
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              top_n: int = 10) -> plt.Figure:
        """Plot feature importance bar chart."""
        try:
            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot horizontal bar chart
            features = [f[0] for f in sorted_features]
            importance = [f[1] for f in sorted_features]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importance, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            
            # Add labels and title
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top Feature Importance Scores')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def plot_category_importance(self, categorized_importance: Dict[str, float]) -> plt.Figure:
        """Plot category importance pie chart."""
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Prepare data
            categories = list(categorized_importance.keys())
            importance = list(categorized_importance.values())
            colors = [self.colors.get(cat, '#95a5a6') for cat in categories]
            
            # Plot pie chart
            ax.pie(importance, labels=categories, colors=colors, autopct='%1.1f%%',
                  startangle=90)
            ax.axis('equal')
            
            # Add title
            ax.set_title('Category Importance Distribution')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting category importance: {str(e)}")
            raise
    
    def plot_temporal_patterns(self, data: Dict[str, Any]) -> plt.Figure:
        """Plot temporal patterns in the data."""
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot data frequency
            timestamps = []
            for source in ["wearable", "ehr", "survey"]:
                if source in data and "timestamp" in data[source]:
                    timestamps.append(pd.to_datetime(data[source]["timestamp"]))
            
            if timestamps:
                time_diffs = np.diff(sorted(timestamps))
                ax1.hist(time_diffs, bins=30)
                ax1.set_title('Data Collection Frequency')
                ax1.set_xlabel('Time between measurements')
                ax1.set_ylabel('Frequency')
            
            # Plot adherence trends
            if "ehr" in data and "medications" in data["ehr"]:
                meds = data["ehr"]["medications"]
                adherence = [m.get("adherence", 0) for m in meds]
                dates = [pd.to_datetime(m.get("date", "")) for m in meds]
                
                ax2.plot(dates, adherence, marker='o')
                ax2.set_title('Medication Adherence Trend')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Adherence Rate')
                ax2.grid(True)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting temporal patterns: {str(e)}")
            raise
    
    def plot_behavioral_patterns(self, data: Dict[str, Any]) -> plt.Figure:
        """Plot behavioral patterns in the data."""
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot activity patterns
            if "wearable" in data and "activity" in data["wearable"]:
                activity = data["wearable"]["activity"]
                sns.histplot(activity, ax=ax1, bins=30)
                ax1.set_title('Activity Level Distribution')
                ax1.set_xlabel('Activity Level')
                ax1.set_ylabel('Frequency')
            
            # Plot sleep patterns
            if "wearable" in data and "sleep" in data["wearable"]:
                sleep = data["wearable"]["sleep"]
                sleep_duration = [s.get("duration", 0) for s in sleep]
                sleep_quality = [s.get("quality", 0) for s in sleep]
                
                ax2.scatter(sleep_duration, sleep_quality)
                ax2.set_title('Sleep Duration vs Quality')
                ax2.set_xlabel('Sleep Duration (hours)')
                ax2.set_ylabel('Sleep Quality Score')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting behavioral patterns: {str(e)}")
            raise
    
    def create_explanation_dashboard(self, explanation: Dict[str, Any]) -> List[plt.Figure]:
        """Create a complete explanation dashboard."""
        try:
            figures = []
            
            # Plot feature importance
            if "feature_importance" in explanation:
                fig1 = self.plot_feature_importance(explanation["feature_importance"])
                figures.append(fig1)
            
            # Plot category importance
            if "categorized_importance" in explanation:
                fig2 = self.plot_category_importance(explanation["categorized_importance"])
                figures.append(fig2)
            
            # Plot temporal patterns
            if "data" in explanation:
                fig3 = self.plot_temporal_patterns(explanation["data"])
                figures.append(fig3)
            
            # Plot behavioral patterns
            if "data" in explanation:
                fig4 = self.plot_behavioral_patterns(explanation["data"])
                figures.append(fig4)
            
            return figures
            
        except Exception as e:
            logger.error(f"Error creating explanation dashboard: {str(e)}")
            raise 