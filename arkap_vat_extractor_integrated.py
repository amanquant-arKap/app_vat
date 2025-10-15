import streamlit as st
import pandas as pd
import io, re, time, requests, random, string, os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats

# Try importing pyodbc for Access database (optional)
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False


# ============================================================================
# ENHANCED LINKEDIN CONTACT ANALYTICS MODULE
# ============================================================================

class LinkedInContactDB:
    """
    Enhanced Manager for LinkedIn Contact Database (linkedinDB.accdb)
    Provides advanced data loading, clustering, and analytics visualization
    """
    
    def __init__(self):
        self.df = None
        self.db_path = "linkedinDB.accdb"
        self.encoded_df = None
        self.cluster_results = {}
    
    def load_from_access(self, db_path=None):
        """Load data from Access database file"""
        if not PYODBC_AVAILABLE:
            st.error("❌ pyodbc not installed. Install with: pip install pyodbc")
            return False
        
        try:
            path = db_path or self.db_path
            
            with st.spinner(f"📥 Loading {path}..."):
                conn_str = (
                    r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    rf'DBQ={path};'
                )
                
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                tables = [table.table_name for table in cursor.tables(tableType='TABLE')]
                
                if not tables:
                    st.error("❌ No tables found in database")
                    return False
                
                table_name = tables[0]
                st.info(f"📊 Loading table: {table_name}")
                
                query = f"SELECT * FROM [{table_name}]"
                self.df = pd.read_sql(query, conn)
                conn.close()
                
                st.success(f"✅ Loaded {len(self.df)} contacts from {table_name}")
                return True
                
        except Exception as e:
            st.error(f"❌ Error loading Access database: {str(e)}")
            st.info("💡 Make sure linkedinDB.accdb is in the same folder as this app")
            return False
    
    def load_from_upload(self, uploaded_file):
        """Load data from uploaded file (CSV or Excel) as fallback"""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.df = pd.read_csv(uploaded_file)
            else:
                self.df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(self.df)} contacts")
            return True
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return False
    
    def get_clustering_columns(self):
        """Identify columns suitable for clustering analysis"""
        if self.df is None:
            return []
        
        priority_cols = ['region', 'province', 'marginoferror2', 'country', 
                        'city', 'industry', 'position', 'company', 'sector',
                        'department', 'level', 'seniority', 'function']
        
        available_cols = []
        for col in priority_cols:
            matching = [c for c in self.df.columns if col.lower() in c.lower()]
            available_cols.extend(matching)
        
        for col in self.df.columns:
            if col not in available_cols and self.df[col].dtype == 'object':
                available_cols.append(col)
        
        return list(dict.fromkeys(available_cols))
    
    def get_numeric_columns(self):
        """Get numeric columns for metrics"""
        if self.df is None:
            return []
        return [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
    
    def prepare_for_ml_clustering(self, columns_to_use):
        """Prepare data for machine learning clustering algorithms"""
        try:
            subset = self.df[columns_to_use].copy()
            subset = subset.dropna()
            
            if len(subset) == 0:
                st.warning("No complete data rows for selected columns")
                return None, None
            
            encoded_data = pd.DataFrame()
            encoders = {}
            
            for col in columns_to_use:
                if subset[col].dtype == 'object':
                    le = LabelEncoder()
                    encoded_data[col] = le.fit_transform(subset[col].astype(str))
                    encoders[col] = le
                else:
                    encoded_data[col] = subset[col]
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(encoded_data)
            
            return scaled_data, subset.index
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None, None
    
    def perform_kmeans_clustering(self, columns_to_use, n_clusters=5):
        """Perform K-means clustering"""
        scaled_data, valid_indices = self.prepare_for_ml_clustering(columns_to_use)
        
        if scaled_data is None:
            return None
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            cluster_df = self.df.loc[valid_indices].copy()
            cluster_df['Cluster'] = clusters
            
            cluster_stats = cluster_df.groupby('Cluster').agg({
                columns_to_use[0]: 'count'
            }).rename(columns={columns_to_use[0]: 'Count'})
            
            return {
                'clusters': clusters,
                'df': cluster_df,
                'stats': cluster_stats,
                'inertia': kmeans.inertia_,
                'centers': kmeans.cluster_centers_
            }
            
        except Exception as e:
            st.error(f"K-means clustering error: {str(e)}")
            return None
    
    def visualize_advanced_clusters(self, cluster_col, show_stats=True):
        """Create advanced visualizations for clustering dimension"""
        if self.df is None:
            st.warning("No data loaded")
            return
        
        st.subheader(f"📊 Advanced Analysis: {cluster_col}")
        
        clean_df = self.df[self.df[cluster_col].notna()].copy()
        
        if len(clean_df) == 0:
            st.warning(f"No data available for {cluster_col}")
            return
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "📊 Statistics", "🎯 Top Values", "📉 Patterns"])
        
        with tab1:
            cluster_counts = clean_df[cluster_col].value_counts().head(30)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Interactive bar chart
                fig_bar = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': cluster_col, 'y': 'Count'},
                    title=f'Distribution by {cluster_col}',
                    color=cluster_counts.values,
                    color_continuous_scale='Viridis',
                    text=cluster_counts.values
                )
                fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                fig_bar.update_layout(
                    showlegend=False, 
                    xaxis_tickangle=-45,
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Donut chart
                fig_donut = go.Figure(data=[go.Pie(
                    labels=cluster_counts.head(10).index,
                    values=cluster_counts.head(10).values,
                    hole=.4,
                    marker=dict(colors=px.colors.qualitative.Set3)
                )])
                fig_donut.update_layout(
                    title=f'Top 10 {cluster_col}',
                    height=500
                )
                st.plotly_chart(fig_donut, use_container_width=True)
        
        with tab2:
            # Statistical analysis
            st.subheader("📊 Statistical Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(clean_df):,}")
            with col2:
                st.metric("Unique Values", f"{clean_df[cluster_col].nunique():,}")
            with col3:
                entropy = stats.entropy(cluster_counts.values)
                st.metric("Distribution Entropy", f"{entropy:.2f}")
            with col4:
                sorted_vals = np.sort(cluster_counts.values)
                n = len(sorted_vals)
                cumsum = np.cumsum(sorted_vals)
                gini = (2 * np.sum((np.arange(1, n+1)) * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
                st.metric("Concentration (Gini)", f"{gini:.3f}")
            
            # Detailed statistics table
            stats_df = pd.DataFrame({
                cluster_col: cluster_counts.index[:20],
                'Count': cluster_counts.values[:20],
                'Percentage': (cluster_counts.values[:20] / len(clean_df) * 100).round(2),
                'Cumulative %': np.cumsum(cluster_counts.values[:20] / len(clean_df) * 100).round(2)
            })
            
            st.dataframe(stats_df, use_container_width=True, height=400)
        
        with tab3:
            # Top values with comparison
            st.subheader("🎯 Top Values Analysis")
            
            top_n = st.slider("Number of top values to show", 5, 30, 15, key=f"top_{cluster_col}")
            
            top_values = cluster_counts.head(top_n)
            
            # Horizontal bar chart
            fig_h = go.Figure(go.Bar(
                x=top_values.values,
                y=top_values.index,
                orientation='h',
                marker=dict(
                    color=top_values.values,
                    colorscale='Blues',
                    showscale=True
                ),
                text=top_values.values,
                textposition='outside'
            ))
            
            fig_h.update_layout(
                title=f'Top {top_n} {cluster_col} by Count',
                xaxis_title='Count',
                yaxis_title=cluster_col,
                height=max(400, top_n * 25),
                showlegend=False
            )
            
            st.plotly_chart(fig_h, use_container_width=True)
            
            # Pareto chart
            cumsum_pct = np.cumsum(cluster_counts.values) / cluster_counts.values.sum() * 100
            
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_pareto.add_trace(
                go.Bar(x=list(range(len(cluster_counts.head(20)))), 
                       y=cluster_counts.head(20).values,
                       name="Count",
                       marker_color='lightblue'),
                secondary_y=False
            )
            
            fig_pareto.add_trace(
                go.Scatter(x=list(range(len(cumsum_pct[:20]))), 
                          y=cumsum_pct[:20],
                          name="Cumulative %",
                          line=dict(color='red', width=2),
                          marker=dict(size=8)),
                secondary_y=True
            )
            
            fig_pareto.update_layout(
                title="Pareto Analysis (80/20 Rule)",
                xaxis_title="Rank",
                height=400
            )
            
            fig_pareto.update_yaxes(title_text="Count", secondary_y=False)
            fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True)
            
            st.plotly_chart(fig_pareto, use_container_width=True)
        
        with tab4:
            # Distribution patterns
            st.subheader("📉 Distribution Patterns")
            
            value_counts = clean_df[cluster_col].value_counts()
            
            # Box plot
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=value_counts.values,
                name="Distribution",
                boxmean='sd',
                marker_color='lightseagreen'
            ))
            
            fig_box.update_layout(
                title="Distribution Shape (Box Plot of Frequencies)",
                yaxis_title="Frequency Count",
                height=400
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Histogram
            fig_hist = px.histogram(
                x=value_counts.values,
                nbins=30,
                title="Histogram of Category Frequencies",
                labels={'x': 'Frequency', 'y': 'Number of Categories'}
            )
            fig_hist.update_traces(marker_color='indianred', marker_line_color='darkred', marker_line_width=1)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    def ml_clustering_interface(self):
        """Interface for ML-based clustering (K-means)"""
        st.subheader("🤖 Machine Learning Clustering")
        
        clustering_cols = self.get_clustering_columns()
        
        if not clustering_cols:
            st.warning("No suitable columns for ML clustering")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_features = st.multiselect(
                "Select features for clustering",
                clustering_cols,
                default=clustering_cols[:min(3, len(clustering_cols))],
                help="Choose 2-5 features for best results"
            )
        
        with col2:
            n_clusters = st.slider("Number of clusters", 2, 10, 5)
        
        if len(selected_features) < 2:
            st.warning("Please select at least 2 features")
            return
        
        if st.button("🚀 Run Clustering", type="primary"):
            with st.spinner("Running clustering algorithm..."):
                results = self.perform_kmeans_clustering(selected_features, n_clusters)
                
                if results:
                    st.success(f"✅ Clustering complete! Found {n_clusters} clusters")
                    
                    tab1, tab2, tab3 = st.tabs(["📊 Cluster Distribution", "📈 Visualization", "📋 Details"])
                    
                    with tab1:
                        cluster_sizes = results['df']['Cluster'].value_counts().sort_index()
                        
                        fig_clusters = px.bar(
                            x=cluster_sizes.index,
                            y=cluster_sizes.values,
                            labels={'x': 'Cluster ID', 'y': 'Number of Contacts'},
                            title='Cluster Size Distribution',
                            text=cluster_sizes.values,
                            color=cluster_sizes.values,
                            color_continuous_scale='Plasma'
                        )
                        fig_clusters.update_traces(textposition='outside')
                        st.plotly_chart(fig_clusters, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Contacts", len(results['df']))
                        with col2:
                            avg_size = len(results['df']) / n_clusters
                            st.metric("Avg Cluster Size", f"{avg_size:.0f}")
                        with col3:
                            st.metric("Inertia", f"{results['inertia']:.2f}")
                    
                    with tab2:
                        scaled_data, _ = self.prepare_for_ml_clustering(selected_features)
                        
                        if scaled_data is not None and len(selected_features) > 2:
                            pca = PCA(n_components=2)
                            pca_data = pca.fit_transform(scaled_data)
                            
                            fig_pca = px.scatter(
                                x=pca_data[:, 0],
                                y=pca_data[:, 1],
                                color=results['clusters'].astype(str),
                                labels={'x': 'First Principal Component', 'y': 'Second Principal Component', 'color': 'Cluster'},
                                title=f'Cluster Visualization (PCA) - Variance Explained: {pca.explained_variance_ratio_.sum():.1%}',
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            fig_pca.update_traces(marker=dict(size=8, opacity=0.6))
                            fig_pca.update_layout(height=600)
                            st.plotly_chart(fig_pca, use_container_width=True)
                            
                            st.info(f"📊 PCA Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
                    
                    with tab3:
                        st.subheader("Cluster Characteristics")
                        
                        for cluster_id in sorted(results['df']['Cluster'].unique()):
                            with st.expander(f"🔹 Cluster {cluster_id} ({len(results['df'][results['df']['Cluster']==cluster_id])} contacts)"):
                                cluster_data = results['df'][results['df']['Cluster']==cluster_id]
                                
                                for feat in selected_features:
                                    if feat in cluster_data.columns:
                                        top_vals = cluster_data[feat].value_counts().head(5)
                                        st.write(f"**{feat}:**")
                                        for val, count in top_vals.items():
                                            pct = count / len(cluster_data) * 100
                                            st.write(f"  • {val}: {count} ({pct:.1f}%)")
                        
                        csv = results['df'].to_csv(index=False)
                        st.download_button(
                            "📥 Download Cluster Assignments",
                            csv,
                            f"clusters_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv"
                        )
    
    def cross_analysis_advanced(self, col1, col2):
        """Enhanced cross-analysis with interactive features"""
        if self.df is None:
            return
        
        st.subheader(f"🔀 Advanced Cross-Analysis: {col1} vs {col2}")
        
        clean_df = self.df[[col1, col2]].dropna()
        
        if len(clean_df) == 0:
            st.warning("No data available for cross-analysis")
            return
        
        crosstab = pd.crosstab(clean_df[col1], clean_df[col2])
        
        # Filtering options
        col_a, col_b = st.columns(2)
        with col_a:
            top_rows = st.slider(f"Top {col1} categories", 5, 30, 15, key="cross_rows")
        with col_b:
            top_cols = st.slider(f"Top {col2} categories", 5, 30, 15, key="cross_cols")
        
        if len(crosstab) > top_rows:
            top_idx = clean_df[col1].value_counts().head(top_rows).index
            crosstab = crosstab.loc[top_idx]
        
        if len(crosstab.columns) > top_cols:
            top_col_idx = clean_df[col2].value_counts().head(top_cols).index
            crosstab = crosstab[top_col_idx]
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "📊 Stacked Bar", "📈 Grouped Bar", "📋 Data Table"])
        
        with tab1:
            fig_heat = px.imshow(
                crosstab,
                labels=dict(x=col2, y=col1, color="Count"),
                title=f'Heatmap: {col1} vs {col2}',
                color_continuous_scale='RdYlBu_r',
                aspect='auto',
                text_auto=True
            )
            fig_heat.update_xaxes(side="bottom")
            fig_heat.update_layout(height=max(400, len(crosstab) * 30))
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Chi-square test
            if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:
                from scipy.stats import chi2_contingency
                chi2, p_value, dof, expected = chi2_contingency(crosstab)
                
                st.info(f"📊 Chi-Square Test: χ²={chi2:.2f}, p-value={p_value:.4f}")
                if p_value < 0.05:
                    st.success("✅ Strong association between variables (p < 0.05)")
                else:
                    st.warning("⚠️ Weak association between variables (p ≥ 0.05)")
        
        with tab2:
            fig_stacked = go.Figure()
            
            for col_name in crosstab.columns:
                fig_stacked.add_trace(go.Bar(
                    name=str(col_name),
                    x=crosstab.index,
                    y=crosstab[col_name],
                    text=crosstab[col_name],
                    textposition='inside'
                ))
            
            fig_stacked.update_layout(
                barmode='stack',
                title=f'Stacked Distribution: {col1} by {col2}',
                xaxis_title=col1,
                yaxis_title='Count',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_stacked, use_container_width=True)
        
        with tab3:
            fig_grouped = go.Figure()
            
            for col_name in crosstab.columns:
                fig_grouped.add_trace(go.Bar(
                    name=str(col_name),
                    x=crosstab.index,
                    y=crosstab[col_name],
                    text=crosstab[col_name],
                    textposition='outside'
                ))
            
            fig_grouped.update_layout(
                barmode='group',
                title=f'Grouped Comparison: {col1} by {col2}',
                xaxis_title=col1,
                yaxis_title='Count',
                height=500
            )
            
            st.plotly_chart(fig_grouped, use_container_width=True)
        
        with tab4:
            st.subheader("📊 Detailed Cross-Tabulation")
            
            crosstab_with_totals = crosstab.copy()
            crosstab_with_totals['Total'] = crosstab_with_totals.sum(axis=1)
            crosstab_with_totals.loc['Total'] = crosstab_with_totals.sum()
            
            st.dataframe(crosstab_with_totals, use_container_width=True)
            
            st.subheader("📊 Percentage View (Row-wise)")
            crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
            st.dataframe(crosstab_pct.round(2), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                csv = crosstab_with_totals.to_csv()
                st.download_button("📥 Download Counts", csv, "crosstab_counts.csv", "text/csv")
            with col2:
                csv_pct = crosstab_pct.to_csv()
                st.download_button("📥 Download Percentages", csv_pct, "crosstab_percentages.csv", "text/csv")
    
    def distribution_analysis(self):
        """Comprehensive distribution analysis across multiple dimensions"""
        st.subheader("📊 Multi-Dimensional Distribution Analysis")
        
        cluster_cols = self.get_clustering_columns()
        
        if not cluster_cols:
            st.warning("No suitable columns for analysis")
            return
        
        selected_cols = st.multiselect(
            "Select dimensions to compare",
            cluster_cols,
            default=cluster_cols[:min(4, len(cluster_cols))],
            help="Choose 2-4 dimensions for comparison"
        )
        
        if len(selected_cols) < 2:
            st.info("Select at least 2 dimensions to compare distributions")
            return
        
        if st.button("📈 Analyze Distributions", type="primary"):
            n_dims = len(selected_cols)
            fig = make_subplots(
                rows=(n_dims + 1) // 2,
                cols=2,
                subplot_titles=[f"{col} Distribution" for col in selected_cols],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            for idx, col in enumerate(selected_cols):
                row = (idx // 2) + 1
                col_pos = (idx % 2) + 1
                
                value_counts = self.df[col].value_counts().head(10)
                
                fig.add_trace(
                    go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        name=col,
                        marker_color=px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)],
                        showlegend=False
                    ),
                    row=row,
                    col=col_pos
                )
            
            fig.update_layout(
                height=300 * ((n_dims + 1) // 2),
                title_text="Multi-Dimensional Distribution Comparison",
                showlegend=False
            )
            
            fig.update_xaxes(tickangle=-45)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparative statistics
            st.subheader("📊 Comparative Statistics")
            
            stats_data = []
            for col in selected_cols:
                clean_data = self.df[col].dropna()
                stats_data.append({
                    'Dimension': col,
                    'Total Records': len(clean_data),
                    'Unique Values': clean_data.nunique(),
                    'Most Common': clean_data.mode()[0] if len(clean_data.mode()) > 0 else 'N/A',
                    'Most Common Count': clean_data.value_counts().iloc[0] if len(clean_data) > 0 else 0,
                    'Most Common %': f"{(clean_data.value_counts().iloc[0] / len(clean_data) * 100):.1f}%" if len(clean_data) > 0 else '0%'
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
    
    def interactive_filters(self):
        """Interactive filtering and drill-down interface"""
        st.subheader("🔍 Interactive Data Explorer")
        
        cluster_cols = self.get_clustering_columns()
        
        if not cluster_cols:
            st.warning("No columns available for filtering")
            return
        
        st.write("**Apply Filters:**")
        
        filters = {}
        filter_cols = st.multiselect(
            "Select dimensions to filter",
            cluster_cols,
            default=[],
            help="Choose dimensions to create filters"
        )
        
        if filter_cols:
            cols_per_row = 3
            for i in range(0, len(filter_cols), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col_name in enumerate(filter_cols[i:i+cols_per_row]):
                    with cols[j]:
                        unique_vals = self.df[col_name].dropna().unique()
                        selected = st.multiselect(
                            f"{col_name}",
                            options=sorted(unique_vals.astype(str)),
                            key=f"filter_{col_name}"
                        )
                        if selected:
                            filters[col_name] = selected
            
            if filters:
                filtered_df = self.df.copy()
                for col, values in filters.items():
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]
                
                st.success(f"✅ Filters applied: {len(filtered_df):,} records match (from {len(self.df):,} total)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Filtered Records", f"{len(filtered_df):,}")
                with col2:
                    pct = (len(filtered_df) / len(self.df) * 100) if len(self.df) > 0 else 0
                    st.metric("% of Total", f"{pct:.1f}%")
                with col3:
                    st.metric("Active Filters", len(filters))
                
                with st.expander("📋 View Filtered Data"):
                    st.dataframe(filtered_df, use_container_width=True)
                
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Filtered Data",
                    csv,
                    f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )


# ============================================================================
# ENHANCED CONTACT ANALYTICS INTERFACE
# ============================================================================

def show_contact_analytics():
    """Display Enhanced LinkedIn Contact Analytics Interface"""
    st.title("👥 LinkedIn Contact Analytics")
    
    if st.session_state.linkedin_db is None:
        st.session_state.linkedin_db = LinkedInContactDB()
    
    linkedin_db = st.session_state.linkedin_db
    
    # Load data section
    if linkedin_db.df is None:
        st.header("📥 Load Contact Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("From Access Database")
            if not PYODBC_AVAILABLE:
                st.warning("⚠️ pyodbc not installed")
                st.code("pip install pyodbc")
            else:
                db_path = st.text_input("Database path", value="linkedinDB.accdb")
                if st.button("📂 Load from Access", type="primary"):
                    linkedin_db.load_from_access(db_path)
        
        with col2:
            st.subheader("From File Upload")
            uploaded = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
            if uploaded:
                if st.button("📤 Load Uploaded File", type="primary"):
                    linkedin_db.load_from_upload(uploaded)
        
        if st.button("⬅️ Back to Main Menu"):
            st.session_state.data_type = None
            st.rerun()
        
        return
    
    # Data loaded - show enhanced analytics
    st.success(f"✅ {len(linkedin_db.df)} contacts loaded")
    
    # Show data preview
    with st.expander("👀 Preview Data"):
        st.dataframe(linkedin_db.df.head(20), use_container_width=True)
    
    # Enhanced analytics tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Single Cluster", 
        "🤖 ML Clustering",
        "🔀 Cross-Analysis", 
        "📊 Multi-Distribution",
        "🔍 Interactive Filters",
        "⚙️ Overview & Settings"
    ])
    
    with tab1:
        st.subheader("Single Dimension Advanced Analysis")
        
        cluster_cols = linkedin_db.get_clustering_columns()
        
        if not cluster_cols:
            st.warning("No suitable clustering columns found")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_cluster = st.selectbox(
                    "Select clustering dimension",
                    cluster_cols,
                    help="Choose a column to analyze"
                )
            
            with col2:
                numeric_cols = linkedin_db.get_numeric_columns()
                metric_col = None
                if numeric_cols:
                    metric_col = st.selectbox("Optional metric", [''] + numeric_cols)
            
            if st.button("🔍 Analyze", type="primary", key="analyze_single"):
                linkedin_db.visualize_advanced_clusters(selected_cluster)
    
    with tab2:
        linkedin_db.ml_clustering_interface()
    
    with tab3:
        st.subheader("Cross-Dimension Analysis")
        
        cluster_cols = linkedin_db.get_clustering_columns()
        
        if len(cluster_cols) < 2:
            st.warning("Need at least 2 clustering columns for cross-analysis")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                dim1 = st.selectbox("First dimension", cluster_cols, key='dim1')
            
            with col2:
                dim2_options = [c for c in cluster_cols if c != dim1]
                dim2 = st.selectbox("Second dimension", dim2_options, key='dim2')
            
            if st.button("🔀 Cross-Analyze", type="primary"):
                linkedin_db.cross_analysis_advanced(dim1, dim2)
    
    with tab4:
        linkedin_db.distribution_analysis()
    
    with tab5:
        linkedin_db.interactive_filters()
    
    with tab6:
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Contacts", f"{len(linkedin_db.df):,}")
        
        with col2:
            st.metric("Total Columns", len(linkedin_db.df.columns))
        
        with col3:
            completeness = (1 - linkedin_db.df.isnull().sum().sum() / (len(linkedin_db.df) * len(linkedin_db.df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Column info
        st.subheader("📋 Column Information")
        
        col_info = pd.DataFrame({
            'Column': linkedin_db.df.columns,
            'Type': linkedin_db.df.dtypes.astype(str),
            'Non-Null': linkedin_db.df.count(),
            'Null %': ((linkedin_db.df.isnull().sum() / len(linkedin_db.df)) * 100).round(2)
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # Quick stats
        st.subheader("🎯 Clustering Columns Summary")
        cluster_cols = linkedin_db.get_clustering_columns()
        
        if cluster_cols:
            summary_data = []
            for col in cluster_cols[:10]:
                summary_data.append({
                    'Column': col,
                    'Unique Values': linkedin_db.df[col].nunique(),
                    'Most Common': str(linkedin_db.df[col].mode()[0]) if len(linkedin_db.df[col].mode()) > 0 else 'N/A'
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Settings
        st.subheader("⚙️ Settings & Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reload Data"):
                st.session_state.linkedin_db = None
                st.rerun()
        
        with col2:
            if st.button("⬅️ Back to Main Menu"):
                st.session_state.data_type = None
                st.rerun()
        
        # Export
        st.subheader("💾 Export Data")
        
        csv = linkedin_db.df.to_csv(index=False)
        
        st.download_button(
            "📥 Download Full Dataset",
            csv,
            f"linkedin_contacts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
