"""
Adherence Analysis Example for All of Us Dataset

This script demonstrates how to use the adherence labeling system with the All of Us
Registered Tier Dataset v8 to analyze clinical trial adherence patterns.

"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Import our modules
import sys
sys.path.append('..')

from data_integration.all_of_us_connector import AllOfUsConnector
from models.adherence_labeler import AllOfUsAdherenceLabeler, AdherenceConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdherenceAnalysisExample:
    """
    Example class demonstrating adherence analysis workflow.
    """
    
    def __init__(self):
        """Initialize the analysis example."""
        self.connector = AllOfUsConnector()
        self.labeler = AllOfUsAdherenceLabeler()
        self.results = {}
        
    async def run_complete_analysis(self):
        """Run the complete adherence analysis workflow."""
        logger.info("Starting complete adherence analysis workflow...")
        
        try:
            # Step 1: Load All of Us data
            logger.info("Step 1: Loading All of Us data...")
            data = await self._load_sample_data()
            
            # Step 2: Create multi-class labels
            logger.info("Step 2: Creating multi-class adherence labels...")
            multi_class_labels = self._create_multi_class_labels(data)
            
            # Step 3: Create state-based labels
            logger.info("Step 3: Creating state-based adherence labels...")
            state_labels = self._create_state_based_labels(data)
            
            # Step 4: Analyze patterns
            logger.info("Step 4: Analyzing adherence patterns...")
            analysis_results = self._analyze_patterns(multi_class_labels, data)
            
            # Step 5: Generate visualizations
            logger.info("Step 5: Generating visualizations...")
            self._create_visualizations(multi_class_labels, state_labels, data, analysis_results)
            
            # Step 6: Export results
            logger.info("Step 6: Exporting results...")
            self._export_results(multi_class_labels, state_labels, analysis_results)
            
            logger.info("Complete adherence analysis workflow finished successfully!")
            
        except Exception as e:
            logger.error(f"Error in analysis workflow: {str(e)}")
            raise
    
    async def _load_sample_data(self) -> dict:
        """Load sample data from All of Us domains."""
        try:
            # Define study parameters
            start_date = datetime.now() - timedelta(days=365)  # 1 year of data
            end_date = datetime.now()
            person_ids = list(range(1000, 1020))  # 20 participants
            
            # Load all relevant domains
            domains = [
                'person',
                'fitbit_activity',
                'fitbit_device',
                'fitbit_heart_rate_level',
                'fitbit_sleep_daily_summary',
                'survey'
            ]
            
            data = await self.connector.load_all_domains(
                person_ids=person_ids,
                start_date=start_date,
                end_date=end_date,
                domains=domains
            )
            
            # Print data summary
            summary = self.connector.get_data_summary(data)
            logger.info("Data Summary:")
            for domain, stats in summary.items():
                logger.info(f"  {domain}: {stats['record_count']} records, {stats['person_count']} persons")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading sample data: {str(e)}")
            raise
    
    def _create_multi_class_labels(self, data: dict) -> pd.DataFrame:
        """Create multi-class adherence labels."""
        try:
            # Create labels for 1-year study period
            labels = self.labeler.create_multi_class_labels(
                person_data=data,
                study_period_days=365
            )
            
            # Print summary
            logger.info("Multi-class Labels Summary:")
            level_counts = labels['adherence_level'].value_counts()
            for level, count in level_counts.items():
                percentage = (count / len(labels)) * 100
                logger.info(f"  {level}: {count} participants ({percentage:.1f}%)")
            
            # Calculate average adherence scores
            avg_scores = labels.groupby('adherence_level')['adherence_score'].mean()
            logger.info("Average Adherence Scores by Level:")
            for level, score in avg_scores.items():
                logger.info(f"  {level}: {score:.3f}")
            
            return labels
            
        except Exception as e:
            logger.error(f"Error creating multi-class labels: {str(e)}")
            raise
    
    def _create_state_based_labels(self, data: dict) -> pd.DataFrame:
        """Create state-based adherence labels."""
        try:
            # Define study period
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
            
            # Create state labels
            state_labels = self.labeler.create_state_based_labels(
                person_data=data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Print summary
            logger.info("State-based Labels Summary:")
            state_counts = state_labels['adherence_state'].value_counts()
            for state, count in state_counts.items():
                percentage = (count / len(state_labels)) * 100
                logger.info(f"  {state}: {count} observations ({percentage:.1f}%)")
            
            return state_labels
            
        except Exception as e:
            logger.error(f"Error creating state-based labels: {str(e)}")
            raise
    
    def _analyze_patterns(self, labels_df: pd.DataFrame, data: dict) -> dict:
        """Analyze adherence patterns."""
        try:
            # Perform comprehensive analysis
            analysis = self.labeler.analyze_adherence_patterns(labels_df, data)
            
            # Print key findings
            logger.info("Adherence Pattern Analysis:")
            
            # Summary statistics
            if 'summary_stats' in analysis:
                stats = analysis['summary_stats']
                logger.info(f"  Total participants: {stats.get('total_participants', 'N/A')}")
                if 'level_distribution' in stats:
                    logger.info("  Level distribution:")
                    for level, count in stats['level_distribution'].items():
                        logger.info(f"    {level}: {count}")
            
            # Demographic analysis
            if 'demographic_analysis' in analysis:
                demo = analysis['demographic_analysis']
                if 'age_analysis' in demo:
                    logger.info("  Age group analysis available")
                if 'gender_analysis' in demo:
                    logger.info("  Gender analysis available")
            
            # Behavioral patterns
            if 'behavioral_patterns' in analysis:
                patterns = analysis['behavioral_patterns']
                if 'activity_by_adherence' in patterns:
                    logger.info("  Activity patterns by adherence level available")
                if 'sleep_by_adherence' in patterns:
                    logger.info("  Sleep patterns by adherence level available")
            
            # Risk factors
            if 'risk_factors' in analysis:
                risks = analysis['risk_factors']
                if 'age_comparison' in risks:
                    age_comp = risks['age_comparison']
                    logger.info(f"  Age difference (high - low adherence): {age_comp.get('age_difference', 'N/A'):.1f} years")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            raise
    
    def _create_visualizations(
        self,
        multi_class_labels: pd.DataFrame,
        state_labels: pd.DataFrame,
        data: dict,
        analysis_results: dict
    ):
        """Create comprehensive visualizations."""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create output directory
            output_dir = Path("adherence_analysis_output")
            output_dir.mkdir(exist_ok=True)
            
            # 1. Multi-class adherence distribution
            self._plot_adherence_distribution(multi_class_labels, output_dir)
            
            # 2. Adherence scores distribution
            self._plot_adherence_scores(multi_class_labels, output_dir)
            
            # 3. State transitions over time
            self._plot_state_transitions(state_labels, output_dir)
            
            # 4. Activity patterns by adherence level
            self._plot_activity_patterns(multi_class_labels, data, output_dir)
            
            # 5. Sleep patterns by adherence level
            self._plot_sleep_patterns(multi_class_labels, data, output_dir)
            
            # 6. Survey completion patterns
            self._plot_survey_patterns(multi_class_labels, data, output_dir)
            
            # 7. Demographic analysis
            self._plot_demographic_analysis(multi_class_labels, data, output_dir)
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def _plot_adherence_distribution(self, labels_df: pd.DataFrame, output_dir: Path):
        """Plot adherence level distribution."""
        plt.figure(figsize=(10, 6))
        
        # Create pie chart
        level_counts = labels_df['adherence_level'].value_counts()
        colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Gold, Red
        
        plt.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Distribution of Adherence Levels', fontsize=16, fontweight='bold')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'adherence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_adherence_scores(self, labels_df: pd.DataFrame, output_dir: Path):
        """Plot adherence scores distribution."""
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall score distribution
        ax1.hist(labels_df['adherence_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Adherence Score')
        ax1.set_ylabel('Number of Participants')
        ax1.set_title('Overall Adherence Score Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Score distribution by level
        for level in ['high', 'medium', 'low']:
            level_data = labels_df[labels_df['adherence_level'] == level]['adherence_score']
            ax2.hist(level_data, bins=10, alpha=0.6, label=level.capitalize())
        
        ax2.set_xlabel('Adherence Score')
        ax2.set_ylabel('Number of Participants')
        ax2.set_title('Adherence Score Distribution by Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'adherence_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_state_transitions(self, state_labels: pd.DataFrame, output_dir: Path):
        """Plot state transitions over time."""
        plt.figure(figsize=(15, 8))
        
        # Sample a few participants for visualization
        sample_participants = state_labels['person_id'].unique()[:5]
        
        for i, person_id in enumerate(sample_participants):
            person_data = state_labels[state_labels['person_id'] == person_id].sort_values('date')
            
            # Create state mapping
            state_map = {'active': 3, 'inactive': 2, 'exit': 1}
            y_values = [state_map[state] for state in person_data['adherence_state']]
            
            plt.plot(person_data['date'], y_values, marker='o', linewidth=2, 
                    label=f'Participant {person_id}', alpha=0.8)
        
        plt.yticks([1, 2, 3], ['Exit', 'Inactive', 'Active'])
        plt.xlabel('Date')
        plt.ylabel('Adherence State')
        plt.title('Adherence State Transitions Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'state_transitions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_activity_patterns(self, labels_df: pd.DataFrame, data: dict, output_dir: Path):
        """Plot activity patterns by adherence level."""
        try:
            fitbit_activity = data.get('fitbit_activity', pd.DataFrame())
            if fitbit_activity.empty:
                logger.warning("No Fitbit activity data available for plotting")
                return
            
            # Merge with labels
            merged_data = labels_df.merge(fitbit_activity, on='person_id', how='left')
            
            plt.figure(figsize=(15, 10))
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Daily steps by adherence level
            for level in ['high', 'medium', 'low']:
                level_data = merged_data[merged_data['adherence_level'] == level]['steps']
                ax1.hist(level_data, bins=20, alpha=0.6, label=level.capitalize())
            
            ax1.set_xlabel('Daily Steps')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Daily Steps Distribution by Adherence Level')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Active minutes by adherence level
            for level in ['high', 'medium', 'low']:
                level_data = merged_data[merged_data['adherence_level'] == level]['very_active_minutes']
                ax2.hist(level_data, bins=20, alpha=0.6, label=level.capitalize())
            
            ax2.set_xlabel('Very Active Minutes')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Very Active Minutes by Adherence Level')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Calories burned by adherence level
            for level in ['high', 'medium', 'low']:
                level_data = merged_data[merged_data['adherence_level'] == level]['calories_out']
                ax3.hist(level_data, bins=20, alpha=0.6, label=level.capitalize())
            
            ax3.set_xlabel('Calories Burned')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Calories Burned by Adherence Level')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Box plot of steps by adherence level
            level_data = [merged_data[merged_data['adherence_level'] == level]['steps'] 
                         for level in ['high', 'medium', 'low']]
            ax4.boxplot(level_data, labels=['High', 'Medium', 'Low'])
            ax4.set_ylabel('Daily Steps')
            ax4.set_title('Steps Distribution by Adherence Level')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'activity_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting activity patterns: {str(e)}")
    
    def _plot_sleep_patterns(self, labels_df: pd.DataFrame, data: dict, output_dir: Path):
        """Plot sleep patterns by adherence level."""
        try:
            fitbit_sleep = data.get('fitbit_sleep_daily_summary', pd.DataFrame())
            if fitbit_sleep.empty:
                logger.warning("No Fitbit sleep data available for plotting")
                return
            
            # Merge with labels
            merged_data = labels_df.merge(fitbit_sleep, on='person_id', how='left')
            
            plt.figure(figsize=(15, 10))
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Sleep duration by adherence level
            for level in ['high', 'medium', 'low']:
                level_data = merged_data[merged_data['adherence_level'] == level]['minute_asleep'] / 60
                ax1.hist(level_data, bins=20, alpha=0.6, label=level.capitalize())
            
            ax1.set_xlabel('Sleep Duration (hours)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Sleep Duration by Adherence Level')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Sleep efficiency by adherence level
            merged_data['sleep_efficiency'] = merged_data['minute_asleep'] / merged_data['minute_in_bed']
            for level in ['high', 'medium', 'low']:
                level_data = merged_data[merged_data['adherence_level'] == level]['sleep_efficiency']
                ax2.hist(level_data, bins=20, alpha=0.6, label=level.capitalize())
            
            ax2.set_xlabel('Sleep Efficiency')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Sleep Efficiency by Adherence Level')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Deep sleep by adherence level
            for level in ['high', 'medium', 'low']:
                level_data = merged_data[merged_data['adherence_level'] == level]['minute_deep'] / 60
                ax3.hist(level_data, bins=20, alpha=0.6, label=level.capitalize())
            
            ax3.set_xlabel('Deep Sleep Duration (hours)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Deep Sleep Duration by Adherence Level')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Box plot of sleep duration by adherence level
            level_data = [merged_data[merged_data['adherence_level'] == level]['minute_asleep'] / 60
                         for level in ['high', 'medium', 'low']]
            ax4.boxplot(level_data, labels=['High', 'Medium', 'Low'])
            ax4.set_ylabel('Sleep Duration (hours)')
            ax4.set_title('Sleep Duration Distribution by Adherence Level')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'sleep_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting sleep patterns: {str(e)}")
    
    def _plot_survey_patterns(self, labels_df: pd.DataFrame, data: dict, output_dir: Path):
        """Plot survey completion patterns."""
        try:
            survey_data = data.get('survey', pd.DataFrame())
            if survey_data.empty:
                logger.warning("No survey data available for plotting")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Count surveys per participant
            survey_counts = survey_data.groupby('person_id').size().reset_index(name='survey_count')
            merged_data = labels_df.merge(survey_counts, on='person_id', how='left')
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Survey count distribution by adherence level
            for level in ['high', 'medium', 'low']:
                level_data = merged_data[merged_data['adherence_level'] == level]['survey_count']
                ax1.hist(level_data, bins=10, alpha=0.6, label=level.capitalize())
            
            ax1.set_xlabel('Number of Surveys Completed')
            ax1.set_ylabel('Number of Participants')
            ax1.set_title('Survey Completion by Adherence Level')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Survey types completed
            survey_types = survey_data['survey'].value_counts()
            ax2.bar(range(len(survey_types)), survey_types.values)
            ax2.set_xlabel('Survey Type')
            ax2.set_ylabel('Number of Responses')
            ax2.set_title('Survey Type Distribution')
            ax2.set_xticks(range(len(survey_types)))
            ax2.set_xticklabels(survey_types.index, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'survey_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting survey patterns: {str(e)}")
    
    def _plot_demographic_analysis(self, labels_df: pd.DataFrame, data: dict, output_dir: Path):
        """Plot demographic analysis."""
        try:
            person_data = data.get('person', pd.DataFrame())
            if person_data.empty:
                logger.warning("No person data available for demographic analysis")
                return
            
            # Merge with labels
            merged_data = labels_df.merge(person_data, on='person_id', how='left')
            
            plt.figure(figsize=(15, 10))
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Age distribution by adherence level
            merged_data['age'] = (pd.Timestamp.now() - pd.to_datetime(merged_data['date_of_birth'])).dt.days / 365.25
            for level in ['high', 'medium', 'low']:
                level_data = merged_data[merged_data['adherence_level'] == level]['age']
                ax1.hist(level_data, bins=15, alpha=0.6, label=level.capitalize())
            
            ax1.set_xlabel('Age (years)')
            ax1.set_ylabel('Number of Participants')
            ax1.set_title('Age Distribution by Adherence Level')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Gender distribution by adherence level
            gender_adherence = pd.crosstab(merged_data['gender'], merged_data['adherence_level'], normalize='index')
            gender_adherence.plot(kind='bar', ax=ax2)
            ax2.set_xlabel('Gender')
            ax2.set_ylabel('Proportion')
            ax2.set_title('Adherence Level Distribution by Gender')
            ax2.legend(title='Adherence Level')
            ax2.grid(True, alpha=0.3)
            
            # Race distribution by adherence level
            race_adherence = pd.crosstab(merged_data['race'], merged_data['adherence_level'], normalize='index')
            race_adherence.plot(kind='bar', ax=ax3)
            ax3.set_xlabel('Race')
            ax3.set_ylabel('Proportion')
            ax3.set_title('Adherence Level Distribution by Race')
            ax3.legend(title='Adherence Level')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Ethnicity distribution by adherence level
            ethnicity_adherence = pd.crosstab(merged_data['ethnicity'], merged_data['adherence_level'], normalize='index')
            ethnicity_adherence.plot(kind='bar', ax=ax4)
            ax4.set_xlabel('Ethnicity')
            ax4.set_ylabel('Proportion')
            ax4.set_title('Adherence Level Distribution by Ethnicity')
            ax4.legend(title='Adherence Level')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'demographic_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting demographic analysis: {str(e)}")
    
    def _export_results(
        self,
        multi_class_labels: pd.DataFrame,
        state_labels: pd.DataFrame,
        analysis_results: dict
    ):
        """Export analysis results to files."""
        try:
            output_dir = Path("adherence_analysis_output")
            output_dir.mkdir(exist_ok=True)
            
            # Export labels
            multi_class_labels.to_csv(output_dir / 'multi_class_labels.csv', index=False)
            state_labels.to_csv(output_dir / 'state_based_labels.csv', index=False)
            
            # Export analysis results
            with open(output_dir / 'analysis_results.json', 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            # Export labeling configuration
            config_summary = self.labeler.get_labeling_summary()
            with open(output_dir / 'labeling_config.json', 'w') as f:
                json.dump(config_summary, f, indent=2, default=str)
            
            # Create summary report
            self._create_summary_report(multi_class_labels, state_labels, analysis_results, output_dir)
            
            logger.info(f"Results exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise
    
    def _create_summary_report(
        self,
        multi_class_labels: pd.DataFrame,
        state_labels: pd.DataFrame,
        analysis_results: dict,
        output_dir: Path
    ):
        """Create a summary report of the analysis."""
        try:
            report = []
            report.append("ADHERENCE ANALYSIS SUMMARY REPORT")
            report.append("=" * 50)
            report.append("")
            
            # Multi-class labels summary
            report.append("MULTI-CLASS ADHERENCE LABELS")
            report.append("-" * 30)
            level_counts = multi_class_labels['adherence_level'].value_counts()
            for level, count in level_counts.items():
                percentage = (count / len(multi_class_labels)) * 100
                report.append(f"{level.capitalize()}: {count} participants ({percentage:.1f}%)")
            
            report.append("")
            avg_scores = multi_class_labels.groupby('adherence_level')['adherence_score'].mean()
            report.append("Average Adherence Scores:")
            for level, score in avg_scores.items():
                report.append(f"  {level.capitalize()}: {score:.3f}")
            
            report.append("")
            
            # State-based labels summary
            report.append("STATE-BASED ADHERENCE LABELS")
            report.append("-" * 30)
            state_counts = state_labels['adherence_state'].value_counts()
            for state, count in state_counts.items():
                percentage = (count / len(state_labels)) * 100
                report.append(f"{state.capitalize()}: {count} observations ({percentage:.1f}%)")
            
            report.append("")
            
            # Key findings
            report.append("KEY FINDINGS")
            report.append("-" * 15)
            
            if 'summary_stats' in analysis_results:
                stats = analysis_results['summary_stats']
                report.append(f"Total participants analyzed: {stats.get('total_participants', 'N/A')}")
            
            if 'risk_factors' in analysis_results:
                risks = analysis_results['risk_factors']
                if 'age_comparison' in risks:
                    age_comp = risks['age_comparison']
                    report.append(f"Age difference (high - low adherence): {age_comp.get('age_difference', 'N/A'):.1f} years")
            
            report.append("")
            report.append("RECOMMENDATIONS")
            report.append("-" * 15)
            report.append("1. Focus interventions on participants with low adherence scores")
            report.append("2. Monitor state transitions to identify early warning signs")
            report.append("3. Consider demographic factors in intervention design")
            report.append("4. Use behavioral patterns to personalize engagement strategies")
            
            # Write report
            with open(output_dir / 'summary_report.txt', 'w') as f:
                f.write('\n'.join(report))
            
        except Exception as e:
            logger.error(f"Error creating summary report: {str(e)}")

async def main():
    """Main function to run the adherence analysis example."""
    try:
        # Create and run the analysis
        example = AdherenceAnalysisExample()
        await example.run_complete_analysis()
        
        print("\n" + "="*60)
        print("ADHERENCE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nCheck the 'adherence_analysis_output' directory for:")
        print("- Multi-class and state-based adherence labels")
        print("- Comprehensive visualizations")
        print("- Analysis results and summary report")
        print("- Configuration details")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 