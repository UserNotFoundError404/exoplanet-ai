import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import base64
import io

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """
    Comprehensive visualization engine for NASA exoplanet data analysis.
    Creates interactive charts and research-focused visualizations.
    """
    
    def __init__(self):
        # Set default theme for research-focused visuals
        self.default_theme = {
            'paper_bgcolor': '#f8f9fa',
            'plot_bgcolor': 'white',
            'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#2c3e50'},
            'colorway': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
        }
    
    def create_exoplanet_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive overview visualizations."""
        try:
            # Main metrics
            total_planets = len(data)
            unique_systems = data['hostname'].nunique() if 'hostname' in data.columns else 0
            data_sources = data['data_source'].nunique() if 'data_source' in data.columns else 0
            
            # Planet type distribution
            type_counts = data['exoplanet_type'].value_counts() if 'exoplanet_type' in data.columns else pd.Series()
            
            fig_overview = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Planet Types Distribution', 'Data Sources', 'Discovery Timeline',
                              'Radius vs Mass', 'Orbital Period Distribution', 'Host Star Properties'),
                specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "histogram"}, {"type": "scatter"}]]
            )
            
            # Planet types pie chart
            if not type_counts.empty:
                fig_overview.add_trace(
                    go.Pie(labels=type_counts.index, values=type_counts.values, name="Planet Types"),
                    row=1, col=1
                )
            
            # Data sources bar chart
            if 'data_source' in data.columns:
                source_counts = data['data_source'].value_counts()
                fig_overview.add_trace(
                    go.Bar(x=source_counts.index, y=source_counts.values, name="Data Sources"),
                    row=1, col=2
                )
            
            # Discovery timeline
            if 'disc_year' in data.columns:
                year_data = data['disc_year'].dropna()
                if not year_data.empty:
                    fig_overview.add_trace(
                        go.Histogram(x=year_data, nbinsx=20, name="Discoveries by Year"),
                        row=1, col=3
                    )
            
            # Mass vs Radius scatter
            if 'pl_rade' in data.columns and 'pl_masse' in data.columns:
                valid_data = data.dropna(subset=['pl_rade', 'pl_masse'])
                if not valid_data.empty:
                    fig_overview.add_trace(
                        go.Scatter(
                            x=valid_data['pl_rade'], 
                            y=valid_data['pl_masse'],
                            mode='markers',
                            marker=dict(size=5, opacity=0.6),
                            name="Radius vs Mass"
                        ),
                        row=2, col=1
                    )
            
            # Orbital period distribution
            if 'pl_orbper' in data.columns:
                period_data = data['pl_orbper'].dropna()
                if not period_data.empty:
                    fig_overview.add_trace(
                        go.Histogram(x=np.log10(period_data), nbinsx=30, name="Log(Orbital Period)"),
                        row=2, col=2
                    )
            
            # Host star temperature vs mass
            if 'st_teff' in data.columns and 'st_mass' in data.columns:
                valid_data = data.dropna(subset=['st_teff', 'st_mass'])
                if not valid_data.empty:
                    fig_overview.add_trace(
                        go.Scatter(
                            x=valid_data['st_teff'], 
                            y=valid_data['st_mass'],
                            mode='markers',
                            marker=dict(size=5, opacity=0.6),
                            name="Stellar Temp vs Mass"
                        ),
                        row=2, col=3
                    )
            
            fig_overview.update_layout(
                title="Exoplanet Data Overview Dashboard",
                height=800,
                showlegend=False,
                **self.default_theme
            )
            
            return {
                'type': 'plotly',
                'data': fig_overview.to_json(),
                'summary': {
                    'total_planets': total_planets,
                    'unique_systems': unique_systems,
                    'data_sources': data_sources,
                    'most_common_type': type_counts.index[0] if not type_counts.empty else 'Unknown'
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating exoplanet overview: {str(e)}")
            return {'type': 'error', 'message': str(e)}
    
    def create_light_curve(self, 
                          time: List[float], 
                          flux: List[float], 
                          flux_err: Optional[List[float]] = None,
                          target_name: str = "Unknown",
                          mission: str = "TESS") -> Dict[str, Any]:
        """Create interactive light curve visualization."""
        try:
            fig = go.Figure()
            
            # Main light curve
            if flux_err:
                fig.add_trace(go.Scatter(
                    x=time,
                    y=flux,
                    error_y=dict(type='data', array=flux_err, visible=True),
                    mode='markers',
                    marker=dict(size=2, color='#3498db'),
                    name=f'{mission} Light Curve',
                    hovertemplate='<b>Time:</b> %{x:.4f}<br><b>Flux:</b> %{y:.6f}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=time,
                    y=flux,
                    mode='markers',
                    marker=dict(size=2, color='#3498db'),
                    name=f'{mission} Light Curve',
                    hovertemplate='<b>Time:</b> %{x:.4f}<br><b>Flux:</b> %{y:.6f}<extra></extra>'
                ))
            
            # Add trend line
            z = np.polyfit(time, flux, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=time,
                y=p(time),
                mode='lines',
                line=dict(color='#e74c3c', width=2),
                name='Trend Line'
            ))
            
            fig.update_layout(
                title=f"Light Curve Analysis - {target_name}",
                xaxis_title="Time (Days)",
                yaxis_title="Normalized Flux",
                height=500,
                **self.default_theme
            )
            
            # Calculate statistics
            flux_array = np.array(flux)
            stats = {
                'mean_flux': float(np.mean(flux_array)),
                'std_flux': float(np.std(flux_array)),
                'min_flux': float(np.min(flux_array)),
                'max_flux': float(np.max(flux_array)),
                'variability': float(np.std(flux_array) / np.mean(flux_array) * 100)
            }
            
            return {
                'type': 'plotly',
                'data': fig.to_json(),
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Error creating light curve: {str(e)}")
            return {'type': 'error', 'message': str(e)}
    
    def create_transit_analysis(self, 
                               light_curve_data: Dict[str, Any],
                               transit_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create transit analysis visualization."""
        try:
            time = np.array(light_curve_data['time'])
            flux = np.array(light_curve_data['flux'])
            
            # Generate synthetic transit model
            period = transit_params.get('period', 10.0)
            depth = transit_params.get('depth', 1000) / 1e6  # Convert from ppm
            duration = transit_params.get('duration', 4.0) / 24  # Convert hours to days
            epoch = transit_params.get('epoch', time[0])
            
            # Find transit events
            transit_times = []
            t = epoch
            while t < time[-1]:
                if t >= time[0]:
                    transit_times.append(t)
                t += period
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Full Light Curve', 'Phased Light Curve', 
                              'Individual Transits', 'Transit Parameters'),
                specs=[[{"colspan": 2}, None],
                       [{}, {"type": "table"}]]
            )
            
            # Full light curve with transit markers
            fig.add_trace(
                go.Scatter(
                    x=time, y=flux,
                    mode='markers',
                    marker=dict(size=2, color='#3498db'),
                    name='Light Curve'
                ),
                row=1, col=1
            )
            
            # Mark transit events
            for t_transit in transit_times:
                fig.add_vline(x=t_transit, line=dict(color='red', width=1, dash='dash'), row=1, col=1)
            
            # Phased light curve
            if len(transit_times) > 0:
                # Phase fold the data
                phase = ((time - epoch) % period) / period
                phase[phase > 0.5] -= 1  # Center on transit
                
                fig.add_trace(
                    go.Scatter(
                        x=phase, y=flux,
                        mode='markers',
                        marker=dict(size=3, color='#e74c3c'),
                        name='Phased Data'
                    ),
                    row=2, col=1
                )
                
                # Add transit model
                phase_model = np.linspace(-0.5, 0.5, 1000)
                transit_model = np.ones_like(phase_model)
                in_transit = np.abs(phase_model) < (duration / period / 2)
                transit_model[in_transit] = 1 - depth
                
                fig.add_trace(
                    go.Scatter(
                        x=phase_model, y=transit_model,
                        mode='lines',
                        line=dict(color='green', width=3),
                        name='Transit Model'
                    ),
                    row=2, col=1
                )
            
            # Parameters table
            param_table = go.Table(
                header=dict(values=['Parameter', 'Value', 'Unit']),
                cells=dict(values=[
                    ['Period', 'Duration', 'Depth', 'Epoch', 'Impact Parameter'],
                    [f"{period:.3f}", f"{duration*24:.2f}", f"{depth*1e6:.1f}", 
                     f"{epoch:.3f}", f"{transit_params.get('impact_parameter', 0.5):.2f}"],
                    ['days', 'hours', 'ppm', 'BJD', '']
                ])
            )
            fig.add_trace(param_table, row=2, col=2)
            
            fig.update_layout(
                title=f"Transit Analysis - {light_curve_data.get('target_name', 'Unknown')}",
                height=800,
                **self.default_theme
            )
            
            return {
                'type': 'plotly',
                'data': fig.to_json(),
                'transit_count': len(transit_times),
                'parameters': transit_params
            }
            
        except Exception as e:
            logger.error(f"Error creating transit analysis: {str(e)}")
            return {'type': 'error', 'message': str(e)}
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation heatmap for numerical features."""
        try:
            # Select numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove columns with too many missing values
            valid_cols = []
            for col in numerical_cols:
                if data[col].notna().sum() / len(data) > 0.5:  # At least 50% non-null
                    valid_cols.append(col)
            
            if len(valid_cols) < 2:
                return {'type': 'error', 'message': 'Insufficient numerical data for correlation analysis'}
            
            # Calculate correlation matrix
            corr_matrix = data[valid_cols].corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Feature Correlation Matrix",
                width=600,
                height=600,
                **self.default_theme
            )
            
            return {
                'type': 'plotly',
                'data': fig.to_json(),
                'features_analyzed': len(valid_cols)
            }
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return {'type': 'error', 'message': str(e)}
    
    def create_model_performance_chart(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create model performance comparison chart."""
        try:
            models = list(performance_data.keys())
            model_types = [performance_data[m]['model_type'] for m in models]
            accuracies = [performance_data[m]['accuracy'] for m in models]
            precisions = [performance_data[m]['precision'] for m in models]
            recalls = [performance_data[m]['recall'] for m in models]
            f1_scores = [performance_data[m]['f1_score'] for m in models]
            
            fig = go.Figure()
            
            # Add metrics as traces
            fig.add_trace(go.Bar(
                name='Accuracy',
                x=model_types,
                y=accuracies,
                marker_color='#3498db'
            ))
            
            fig.add_trace(go.Bar(
                name='Precision',
                x=model_types,
                y=precisions,
                marker_color='#e74c3c'
            ))
            
            fig.add_trace(go.Bar(
                name='Recall',
                x=model_types,
                y=recalls,
                marker_color='#2ecc71'
            ))
            
            fig.add_trace(go.Bar(
                name='F1-Score',
                x=model_types,
                y=f1_scores,
                marker_color='#f39c12'
            ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model Type",
                yaxis_title="Score",
                barmode='group',
                height=500,
                **self.default_theme
            )
            
            return {
                'type': 'plotly',
                'data': fig.to_json(),
                'best_model': model_types[accuracies.index(max(accuracies))]
            }
            
        except Exception as e:
            logger.error(f"Error creating model performance chart: {str(e)}")
            return {'type': 'error', 'message': str(e)}
    
    def create_feature_importance_plot(self, 
                                     feature_importance: Dict[str, float],
                                     model_name: str) -> Dict[str, Any]:
        """Create feature importance visualization."""
        try:
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_features)
            
            fig = go.Figure(go.Bar(
                y=features,
                x=importance,
                orientation='h',
                marker=dict(color='#3498db')
            ))
            
            fig.update_layout(
                title=f"Feature Importance - {model_name}",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(300, len(features) * 25),
                **self.default_theme
            )
            
            return {
                'type': 'plotly',
                'data': fig.to_json(),
                'top_feature': features[0] if features else None
            }
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            return {'type': 'error', 'message': str(e)}
    
    def create_discovery_timeline(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create discovery timeline visualization."""
        try:
            if 'disc_year' not in data.columns:
                return {'type': 'error', 'message': 'Discovery year data not available'}
            
            # Group by year and facility
            yearly_data = data.groupby(['disc_year', 'disc_facility']).size().reset_index(name='count')
            
            fig = px.bar(
                yearly_data, 
                x='disc_year', 
                y='count',
                color='disc_facility',
                title="Exoplanet Discoveries Timeline by Facility",
                labels={'disc_year': 'Discovery Year', 'count': 'Number of Discoveries'}
            )
            
            fig.update_layout(**self.default_theme)
            
            return {
                'type': 'plotly',
                'data': fig.to_json(),
                'total_years': yearly_data['disc_year'].nunique(),
                'facilities': yearly_data['disc_facility'].nunique()
            }
            
        except Exception as e:
            logger.error(f"Error creating discovery timeline: {str(e)}")
            return {'type': 'error', 'message': str(e)}
    
    def create_planet_classification_plot(self, 
                                        data: pd.DataFrame,
                                        x_feature: str = 'pl_rade',
                                        y_feature: str = 'pl_masse') -> Dict[str, Any]:
        """Create planet classification scatter plot."""
        try:
            if x_feature not in data.columns or y_feature not in data.columns:
                return {'type': 'error', 'message': f'Required features not available: {x_feature}, {y_feature}'}
            
            # Filter valid data
            valid_data = data.dropna(subset=[x_feature, y_feature, 'exoplanet_type'])
            
            if len(valid_data) == 0:
                return {'type': 'error', 'message': 'No valid data for classification plot'}
            
            fig = px.scatter(
                valid_data,
                x=x_feature,
                y=y_feature,
                color='exoplanet_type',
                title=f"Exoplanet Classification: {y_feature} vs {x_feature}",
                hover_data=['pl_name', 'hostname'],
                log_x=True if 'rade' in x_feature else False,
                log_y=True if 'masse' in y_feature else False
            )
            
            fig.update_layout(
                height=600,
                **self.default_theme
            )
            
            return {
                'type': 'plotly',
                'data': fig.to_json(),
                'data_points': len(valid_data),
                'planet_types': valid_data['exoplanet_type'].nunique()
            }
            
        except Exception as e:
            logger.error(f"Error creating planet classification plot: {str(e)}")
            return {'type': 'error', 'message': str(e)}