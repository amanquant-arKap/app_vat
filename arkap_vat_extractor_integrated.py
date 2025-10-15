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
# NACE/ARKAP CONVERTER MODULE
# ============================================================================

class NaceArkapConverter:
    """NACE to Arkap Industry converter with robust fallback strategy"""

    def __init__(self):
        self.lookups = None
        self.enabled = False
        self.mapping_url = None

    def load_mapping_from_url(self, url):
        """Load NACE mapping from Dropbox or URL with timeout and fallback"""
        try:
            if not url:
                return False

            download_url = url.replace('dl=0', 'dl=1').replace('www.dropbox.com', 'dl.dropboxusercontent.com')

            with st.spinner("📥 Loading NACE mapping..."):
                response = requests.get(download_url, timeout=15)
                response.raise_for_status()

                df = pd.read_excel(io.BytesIO(response.content))

                if len(df) > 0:
                    self.lookups = self._create_lookups(df)
                    self.enabled = True
                    st.success(f"✅ NACE mapping loaded: {len(df)} entries")
                    return True
                else:
                    st.warning("⚠️ NACE mapping file is empty - continuing without conversion")
                    return False

        except requests.Timeout:
            st.warning("⚠️ NACE mapping load timeout - continuing without conversion")
            return False
        except requests.RequestException as e:
            st.warning(f"⚠️ NACE mapping unavailable - continuing without conversion")
            return False
        except Exception as e:
            st.warning(f"⚠️ Could not load NACE mapping - continuing without conversion")
            return False

    def _create_lookups(self, df):
        """Create lookup dictionaries from mapping dataframe"""
        try:
            nace_code_lookup = {}
            ateco_code_lookup = {}

            for idx, row in df.iterrows():
                record = {
                    'nace_category_code': str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else '',
                    'nace_category_title': str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else '',
                    'ateco_category_code': str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else '',
                    'ateco_category_title': str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else '',
                    'nace_subcat_code': str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else '',
                    'nace_subcat_title': str(row.iloc[5]).strip() if pd.notna(row.iloc[5]) else '',
                    'ateco_subcat_code': str(row.iloc[6]).strip() if pd.notna(row.iloc[6]) else '',
                    'ateco_subcat_title': str(row.iloc[7]).strip() if pd.notna(row.iloc[7]) else '',
                    'arkap_industry': str(row.iloc[8]).strip() if pd.notna(row.iloc[8]) else '',
                    'arkap_subindustry': str(row.iloc[9]).strip() if pd.notna(row.iloc[9]) else ''
                }

                if record['nace_subcat_code'] and record['nace_subcat_code'] != 'nan':
                    key = record['nace_subcat_code'].lower().strip()
                    nace_code_lookup[key] = record

                if record['ateco_subcat_code'] and record['ateco_subcat_code'] != 'nan':
                    key = record['ateco_subcat_code'].lower().strip()
                    ateco_code_lookup[key] = record

            return {
                'nace_code': nace_code_lookup,
                'ateco_code': ateco_code_lookup
            }
        except Exception as e:
            st.warning(f"⚠️ Error creating NACE lookups - continuing without conversion")
            return None

    def convert_nace_code(self, nace_code):
        """Convert NACE code to Arkap Industry classification"""
        if not self.enabled or not self.lookups or not nace_code:
            return None

        try:
            input_clean = str(nace_code).strip().lower()

            if input_clean in self.lookups['nace_code']:
                result = self.lookups['nace_code'][input_clean]
                return {
                    'arkap_industry': result['arkap_industry'],
                    'arkap_subindustry': result['arkap_subindustry'],
                    'nace_category': result['nace_category_title'],
                    'nace_subcategory': result['nace_subcat_title'],
                    'match_type': 'NACE (Exact)'
                }

            if input_clean in self.lookups['ateco_code']:
                result = self.lookups['ateco_code'][input_clean]
                return {
                    'arkap_industry': result['arkap_industry'],
                    'arkap_subindustry': result['arkap_subindustry'],
                    'ateco_category': result['ateco_category_title'],
                    'ateco_subcategory': result['ateco_subcat_title'],
                    'match_type': 'ATECO (Exact)'
                }

            for key, value in self.lookups['nace_code'].items():
                if key.startswith(input_clean) or input_clean.startswith(key):
                    return {
                        'arkap_industry': value['arkap_industry'],
                        'arkap_subindustry': value['arkap_subindustry'],
                        'nace_category': value['nace_category_title'],
                        'nace_subcategory': value['nace_subcat_title'],
                        'match_type': 'NACE (Partial)'
                    }

            return None

        except Exception as e:
            return None


# ============================================================================
# COMPANY EXTRACTION CODE
# ============================================================================

def get_dropbox_download_link(shared_link):
    if 'dropbox.com' in shared_link:
        return shared_link.replace('dl=0', 'dl=1').replace('www.dropbox.com', 'dl.dropboxusercontent.com')
    return shared_link

def load_database_from_dropbox():
    try:
        if 'DROPBOX_FILE_URL' in st.secrets:
            dropbox_url = st.secrets["DROPBOX_FILE_URL"]
        else:
            st.warning("⚠️ Add DROPBOX_FILE_URL to Streamlit Secrets")
            return None

        download_url = get_dropbox_download_link(dropbox_url)

        with st.spinner("📥 Downloading database..."):
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            df = pd.read_excel(io.BytesIO(response.content))
            st.success(f"✅ Database downloaded: {len(df)} companies")
            return df
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")
        return None

ALLOWED_DOMAIN = "@arkap.ch"
CODE_EXPIRY_MINUTES = 10
SESSION_TIMEOUT_MINUTES = 60
COUNTRY_CODES = {'AT': 'Austria', 'CH': 'Switzerland', 'DE': 'Germany', 'FR': 'France', 'GB': 'United Kingdom', 'IT': 'Italy', 'LU': 'Luxembourg', 'NL': 'Netherlands', 'PT': 'Portugal'}

def safe_format(value, fmt="{:,.0f}", pre="", suf="", default="N/A"):
    if pd.isna(value) or value is None or value == '': return default
    try:
        if isinstance(value, str):
            v = value.replace(',', '').replace(' ', '').replace('€', '').replace('k', '').strip()
            if not v or v == '-': return default
            value = float(v)
        return f"{pre}{fmt.format(float(value))}{suf}"
    except: return str(value) if value else default

class CompanyDatabase:
    """OPTIMIZED: Uses vectorized operations"""
    def __init__(self, df=None):
        self.db = None
        self.name_idx = {}
        self.vat_idx = {}
        self.country_idx = {}

        if df is not None:
            self._init(df)

    def _init(self, df):
        """Optimized database initialization"""
        try:
            with st.spinner("🔧 Preparing database..."):
                mapping = {}
                for col in df.columns:
                    c = col.lower()
                    if 'company' in c and 'name' in c: 
                        mapping[col] = 'Company Name'
                    elif 'vat' in c and 'code' in c: 
                        mapping[col] = 'VAT Code'
                    elif 'national' in c and 'id' in c: 
                        mapping[col] = 'National ID'
                    elif 'fiscal' in c: 
                        mapping[col] = 'Fiscal Code'
                    elif 'country' in c and 'code' in c: 
                        mapping[col] = 'Country Code'
                    elif 'nace' in c: 
                        mapping[col] = 'Nace Code'
                    elif 'last' in c and 'yr' in c: 
                        mapping[col] = 'Last Yr'
                    elif 'production' in c: 
                        mapping[col] = 'Value of production (th)'
                    elif 'employee' in c: 
                        mapping[col] = 'Employees'
                    elif 'ebitda' in c: 
                        mapping[col] = 'Ebitda (th)'
                    elif 'pfn' in c: 
                        mapping[col] = 'PFN (th)'

                self.db = df.rename(columns=mapping)

                if 'Company Name' in self.db.columns:
                    name_series = self.db['Company Name'].dropna().astype(str).str.lower().str.strip()
                    for idx, name in name_series.items():
                        if name:
                            self.name_idx.setdefault(name, []).append(idx)

                if 'VAT Code' in self.db.columns:
                    vat_series = self.db['VAT Code'].dropna().astype(str).str.upper().str.replace(' ', '').str.replace('-', '').str.replace('.', '')
                    for idx, vat in vat_series.items():
                        if vat:
                            self.vat_idx.setdefault(vat, []).append(idx)

                if 'Country Code' in self.db.columns:
                    for cc in self.db['Country Code'].dropna().unique():
                        cc_upper = str(cc).upper()
                        self.country_idx[cc_upper] = self.db[self.db['Country Code'] == cc].index.tolist()

                st.success(f"✅ Database ready: {len(self.db)} companies indexed")

        except Exception as e:
            st.error(f"❌ Database indexing error: {str(e)}")
            self.db = None
            raise

    def search_name(self, name, country=None):
        if self.db is None:
            return None

        k = name.lower().strip()
        if k in self.name_idx:
            idxs = self.name_idx[k]
            if country and country in self.country_idx: 
                idxs = [i for i in idxs if i in self.country_idx[country]]
            return self._extract(self.db.iloc[idxs[0]]) if idxs else None
        return None

    def search_vat(self, vat, country=None):
        if self.db is None:
            return None

        k = str(vat).upper().replace(' ', '').replace('-', '').replace('.', '')
        if k in self.vat_idx:
            idxs = self.vat_idx[k]
            if country and country in self.country_idx: 
                idxs = [i for i in idxs if i in self.country_idx[country]]
            return self._extract(self.db.iloc[idxs[0]]) if idxs else None
        return None

    def _extract(self, row):
        d = {'source': 'database'}
        for f in ['Company Name', 'National ID', 'Fiscal Code', 'VAT Code', 'Country Code', 'Nace Code', 'Last Yr', 'Value of production (th)', 'Employees', 'Ebitda (th)', 'PFN (th)']:
            if f in row.index and pd.notna(row[f]): 
                d[f.lower().replace(' ', '_').replace('(', '').replace(')', '')] = row[f]
        return d

class AuthenticationManager:
    def __init__(self):
        for k in ['auth_codes', 'authenticated', 'user_email', 'auth_time', 'company_db', 'search_mode', 'nace_converter', 'data_type', 'linkedin_db']:
            if k not in st.session_state: 
                st.session_state[k] = {} if k == 'auth_codes' else (False if k == 'authenticated' else ("" if k == 'user_email' else None))

    def is_valid_email(self, e): 
        return re.match(r'^[\w.+-]+@[\w.-]+\.[\w]+$', e) and e.lower().endswith(ALLOWED_DOMAIN.lower())

    def gen_code(self): 
        return ''.join(random.choices(string.digits, k=6))

    def store_code(self, e, c): 
        st.session_state.auth_codes[e] = {'code': c, 'timestamp': datetime.now(), 'attempts': 0}

    def verify(self, e, c):
        if e not in st.session_state.auth_codes: 
            return False, "No code"
        d = st.session_state.auth_codes[e]
        if datetime.now() - d['timestamp'] > timedelta(minutes=CODE_EXPIRY_MINUTES): 
            del st.session_state.auth_codes[e]
            return False, "Expired"
        if d['attempts'] >= 3: 
            del st.session_state.auth_codes[e]
            return False, "Too many"
        if d['code'] == c: 
            st.session_state.authenticated, st.session_state.user_email, st.session_state.auth_time = True, e, datetime.now()
            del st.session_state.auth_codes[e]
            return True, "Success"
        d['attempts'] += 1
        return False, f"{3-d['attempts']} left"

    def is_valid(self): 
        return st.session_state.authenticated and st.session_state.auth_time and datetime.now() - st.session_state.auth_time <= timedelta(minutes=SESSION_TIMEOUT_MINUTES)

    def logout(self): 
        st.session_state.authenticated, st.session_state.user_email, st.session_state.auth_time = False, "", None

class EnhancedUKExtractor:
    """UK Company Number Extractor"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.patterns = [
            r'Company\s+number[\s:]*([0-9]{8})',
            r'(?:Company|Co\.|Ltd\.)\s+(?:No\.|Number)[\s:]*([0-9]{8})',
            r'(?:Registered|Registration)\s+(?:No\.|Number)[\s:]*([0-9]{8})',
            r'([0-9]{8})\s*(?:Company|Registered)',
            r'\b([0-9]{8})\b'
        ]

    def process(self, name, url=None):
        r = {'company_name': name, 'website': url or '', 'status': 'Not Found', 'source': 'web'}
        if url:
            try:
                resp = self.session.get(url, timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    for pattern in self.patterns:
                        for match in re.finditer(pattern, resp.text, re.I):
                            code = re.sub(r'[^0-9]', '', match.group(1) if match.lastindex else match.group(0))
                            if len(code) == 8 and code[0] != '0':
                                r['company_number'] = code
                                r['status'] = 'Found'
                                return r
            except: 
                pass
        return r

class MultiModeExtractor:
    def __init__(self, db=None, use_db=True, nace_converter=None):
        self.db, self.use_db, self.nace_converter = db, use_db, nace_converter
        self.extractors = {'GB': EnhancedUKExtractor()}
        self.patterns = {
            'DE': [
                r'Steuernummer[\s#:]*([0-9]{2,3}/[0-9]{3,4}/[0-9]{4,5})',
                r'Steuer-?Nr\.?[\s#:]*([0-9]{2,3}/[0-9]{3,4}/[0-9]{4,5})',
                r'Handelsregisternummer[\s#:]*([HRA|HRB]{2,3}\s*[0-9]{1,6})',
                r'Umsatzsteuer-?ID[\s#:]*([D|DE]{1,2}[0-9]{9})',
                r'USt-?IdNr\.?[\s#:]*([D|DE]{1,2}[0-9]{9})'
            ],
            'FR': [
                r'SIREN[\s#:]*([0-9]{9})',
                r'(?:N°\s*SIREN|Numéro\s*SIREN)[\s#:]*([0-9]{9})',
                r'SIRET[\s#:]*([0-9]{14})',
                r'TVA[\s#:]*FR([0-9A-Z]{2}[0-9]{9})',
                r'N°\s*TVA[\s#:]*FR([0-9A-Z]{2}[0-9]{9})'
            ],
            'IT': [
                r'P\.?\s*IVA[\s#:]*([0-9]{11})',
                r'Partita\s+IVA[\s#:]*([0-9]{11})',
                r'Codice\s+Fiscale[\s#:]*([A-Z0-9]{11,16})',
                r'C\.?F\.?[\s#:]*([A-Z0-9]{11,16})'
            ],
            'PT': [
                r'NIF[\s#:]*([0-9]{9})',
                r'N\.?I\.?F\.?[\s#:]*([0-9]{9})',
                r'Contribuinte[\s#:]*([0-9]{9})',
                r'NIPC[\s#:]*([0-9]{9})'
            ],
            'NL': [
                r'KvK[\s#:]*([0-9]{8})',
                r'(?:Kamer\s+van\s+Koophandel|K\.v\.K\.?)[\s#:]*([0-9]{8})',
                r'RSIN[\s#:]*([0-9]{9})',
                r'BTW[\s#:]*NL([0-9]{9}B[0-9]{2})',
                r'LEI[\s#:]*([A-Z0-9]{20})'
            ],
            'AT': [
                r'ATU\s*([0-9]{8})',
                r'UID[\s#:]*ATU([0-9]{8})',
                r'Umsatzsteuer-?ID[\s#:]*ATU([0-9]{8})',
                r'FN[\s#:]*([0-9]{6}[a-z])'
            ],
            'CH': [
                r'CHE[\s-]?([0-9]{3})\.?([0-9]{3})\.?([0-9]{3})',
                r'UID[\s#:]*CHE[\s-]?([0-9]{3})\.?([0-9]{3})\.?([0-9]{3})',
                r'CH-ID[\s#:]*CH-([0-9]{3})\.?([0-9]{1})\.?([0-9]{3})\.?([0-9]{3})-?([0-9]{1})'
            ],
            'LU': [
                r'LU\s*([0-9]{8})',
                r'TVA[\s#:]*LU([0-9]{8})',
                r'B([0-9]{6})',
                r'L\.?U\.?R[\s#:]*([0-9]{6})'
            ]
        }

    def process_single(self, name, web, country, vat=None):
        """Process single company"""
        result = None

        if self.use_db and self.db:
            result = self.db.search_name(name, country)
            if result: 
                result['search_method'] = 'DB-Name'
                result['status'] = 'Found'
            elif vat:
                result = self.db.search_vat(vat, country)
                if result: 
                    result['search_method'] = 'DB-VAT'
                    result['status'] = 'Found'

            if not result:
                result = self._web(name, web, country)
                result['search_method'] = 'DB failed → Web'
        else:
            result = self._web(name, web, country)
            result['search_method'] = 'Web only'

        if result and self.nace_converter and 'nace_code' in result:
            conversion = self.nace_converter.convert_nace_code(result['nace_code'])
            if conversion:
                result['arkap_industry'] = conversion.get('arkap_industry', '')
                result['arkap_subindustry'] = conversion.get('arkap_subindustry', '')
                result['nace_conversion_status'] = conversion.get('match_type', 'Converted')

        return result

    def _web(self, name, web, country):
        if country in self.extractors:
            return self.extractors[country].process(name, web)

        r = {'company_name': name, 'website': web, 'country_code': country, 'status': 'Not Found', 'source': 'web'}
        if web and country in self.patterns:
            try:
                resp = requests.get(web, timeout=10)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    text_content = soup.get_text()
                    for pattern in self.patterns[country]:
                        matches = re.finditer(pattern, text_content, re.I)
                        for match in matches:
                            extracted = match.group(1) if match.lastindex else match.group(0)
                            field_name = f'{country.lower()}_code'
                            r[field_name] = extracted.strip()
                            r['status'] = 'Found'
                            return r
            except: 
                pass
        return r

    def process_list(self, df, prog=None):
        """Process list of companies"""
        results = []

        nc = [c for c in df.columns if 'company' in c.lower() or 'name' in c.lower()]
        name_col = nc[0] if nc else df.columns[0]

        wc = [c for c in df.columns if 'website' in c.lower() or 'url' in c.lower()]
        web_col = wc[0] if wc else None

        cc = [c for c in df.columns if 'country' in c.lower()]
        country_col = cc[0] if cc else None

        vc = [c for c in df.columns if 'vat' in c.lower() or 'fiscal' in c.lower()]
        vat_col = vc[0] if vc else None

        for idx, row in df.iterrows():
            if prog: 
                prog(idx+1, len(df))

            name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
            web = str(row[web_col]).strip() if web_col and pd.notna(row[web_col]) else ''
            vat = str(row[vat_col]).strip() if vat_col and pd.notna(row[vat_col]) else None
            country = 'GB'

            if country_col and pd.notna(row[country_col]):
                cv = str(row[country_col]).strip().upper()
                if len(cv) == 2 and cv in COUNTRY_CODES: 
                    country = cv

            result = self.process_single(name, web, country, vat)
            results.append(result)
            time.sleep(0.2)

        return results


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
        
        st.subheader("📋 Column Information")
        
        col_info = pd.DataFrame({
            'Column': linkedin_db.df.columns,
            'Type': linkedin_db.df.dtypes.astype(str),
            'Non-Null': linkedin_db.df.count(),
            'Null %': ((linkedin_db.df.isnull().sum() / len(linkedin_db.df)) * 100).round(2)
        })
        
        st.dataframe(col_info, use_container_width=True)
        
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
        
        st.subheader("💾 Export Data")
        
        csv = linkedin_db.df.to_csv(index=False)
        
        st.download_button(
            "📥 Download Full Dataset",
            csv,
            f"linkedin_contacts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )


# ============================================================================
# UI FUNCTIONS
# ============================================================================

def show_auth(auth):
    st.title("🔐 arKap VAT Extractor")
    st.info("🏢 @arkap.ch only")

    t1, t2 = st.tabs(["📧 Email", "🔑 Code"])

    with t1:
        e = st.text_input("Email")
        if st.button("Send Code", type="primary"):
            if auth.is_valid_email(e):
                c = auth.gen_code()
                auth.store_code(e, c)
                st.success(f"Code: {c}")
            else:
                st.error("Invalid")

    with t2:
        e = st.text_input("Email", key="e2")
        c = st.text_input("Code", max_chars=6)
        if st.button("Verify", type="primary"):
            ok, msg = auth.verify(e, c)
            if ok:
                st.success(msg)
                st.balloons()
                time.sleep(1)
                st.rerun()
            else:
                st.error(msg)

def show_main():
    st.title("🌍 arKap Data Platform")

    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown(f"**User:** {st.session_state.user_email}")
    with c2:
        if st.button("Logout"):
            AuthenticationManager().logout()
            st.rerun()

    st.markdown("---")

    if st.session_state.nace_converter is None:
        st.session_state.nace_converter = NaceArkapConverter()

    with st.expander("🔧 NACE-to-Arkap Mapping (Optional)", expanded=False):
        st.write("Enable industry classification conversion by loading NACE mapping file from Dropbox.")

        nace_url = st.text_input("NACE Mapping Dropbox URL (optional)", 
                                  value=st.secrets.get("NACE_MAPPING_URL", "") if "NACE_MAPPING_URL" in st.secrets else "")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load NACE Mapping"):
                if nace_url:
                    st.session_state.nace_converter.load_mapping_from_url(nace_url)
                else:
                    st.warning("Please provide a Dropbox URL")

        with col2:
            if st.session_state.nace_converter.enabled:
                st.success("✅ NACE Mapping Active")
            else:
                st.info("ℹ️ NACE Mapping Disabled")

    # DATA TYPE SELECTION
    if st.session_state.data_type is None:
        st.header("📊 Select Data Type")
        st.write("Choose what type of data you want to work with:")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🏢 Company Data", type="primary", use_container_width=True):
                st.session_state.data_type = 'company'
                st.rerun()
        with c2:
            if st.button("👥 Contact Data (LinkedIn)", use_container_width=True):
                st.session_state.data_type = 'contact'
                st.rerun()
        
        return
    
    if st.session_state.data_type == 'contact':
        show_contact_analytics()
        return

    # COMPANY DATA FLOW
    st.info(f"📊 Working with: **Company Data**")
    if st.button("🔄 Change Data Type"):
        st.session_state.data_type = None
        st.session_state.search_mode = None
        st.rerun()
    
    st.markdown("---")

    if st.session_state.company_db is None:
        st.header("📊 Database Setup")

        with st.expander("ℹ️ Dropbox Setup", expanded=True):
            st.write("1. Share file on Dropbox → Copy link")
            st.write('2. App Settings → Secrets → Add: DROPBOX_FILE_URL = "your_link"')

        if st.button("📥 Load from Dropbox", type="primary"):
            df = load_database_from_dropbox()
            if df is not None:
                try:
                    st.session_state.company_db = CompanyDatabase(df)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to initialize database: {str(e)}")

        st.markdown("---")

        up = st.file_uploader("Or Upload", type=['xlsx','csv'])
        if up is not None:
            try:
                df = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
                st.info(f"📁 File loaded: {len(df)} rows")
                st.session_state.company_db = CompanyDatabase(df)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to process file: {str(e)}")

        if st.button("⏭️ Web Only"):
            st.session_state.search_mode = 'web'
            st.rerun()

        return

    if st.session_state.search_mode is None:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗄️ DB+Web", type="primary", use_container_width=True):
                st.session_state.search_mode = 'db'
                st.rerun()
        with c2:
            if st.button("🌐 Web Only", use_container_width=True):
                st.session_state.search_mode = 'web'
                st.rerun()
        return

    st.info(f"Mode: {st.session_state.search_mode.upper()}")
    if st.button("Change"):
        st.session_state.search_mode = None
        st.rerun()

    st.markdown("---")

    t1, t2 = st.tabs(["Bulk", "Single"])

    with t1:
        f = st.file_uploader("Company List", type=['csv','xlsx'])
        if f:
            df = pd.read_csv(f) if f.name.endswith('.csv') else pd.read_excel(f)
            st.dataframe(df.head())

            if st.button("Process"):
                ext = MultiModeExtractor(
                    st.session_state.company_db, 
                    st.session_state.search_mode=='db',
                    st.session_state.nace_converter
                )

                p = st.progress(0)
                res = ext.process_list(df, lambda c,t: p.progress(c/t))
                rdf = pd.DataFrame(res)

                st.dataframe(rdf)

                c1,c2,c3 = st.columns(3)
                with c1:
                    st.metric("Total", len(res))
                with c2:
                    st.metric("Found", len([r for r in res if r['status']=='Found']))
                with c3:
                    st.metric("Rate%", f"{len([r for r in res if r['status']=='Found'])/len(res)*100:.1f}")

                csv = io.StringIO()
                rdf.to_csv(csv, index=False)
                st.download_button("Download", csv.getvalue(), f"res_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

    with t2:
        c1,c2 = st.columns(2)
        with c1:
            n = st.text_input("Name")
            w = st.text_input("Website")
            v = st.text_input("VAT")
        with c2:
            co = st.selectbox("Country", list(COUNTRY_CODES.keys()), format_func=lambda x:f"{COUNTRY_CODES[x]} ({x})")

        if st.button("Search") and n:
            ext = MultiModeExtractor(
                st.session_state.company_db, 
                st.session_state.search_mode=='db',
                st.session_state.nace_converter
            )
            r = ext.process_single(n, w, co, v)

            if r['status']=='Found':
                st.success(f"✅ {r.get('search_method')}")

                if r.get('source')=='database':
                    st.subheader("📊 Database Results")
                    c1,c2=st.columns(2)
                    with c1:
                        for k in ['company_name','vat_code']:
                            if k in r: st.write(f"**{k}:** {r[k]}")
                    with c2:
                        for k in ['country_code','nace_code']:
                            if k in r: st.write(f"**{k}:** {r[k]}")

                    if 'arkap_industry' in r or 'arkap_subindustry' in r:
                        st.subheader("🏭 Industry Classification")
                        c1, c2 = st.columns(2)
                        with c1:
                            if 'arkap_industry' in r and r['arkap_industry']:
                                st.write(f"**Arkap Industry:** {r['arkap_industry']}")
                        with c2:
                            if 'arkap_subindustry' in r and r['arkap_subindustry']:
                                st.write(f"**Arkap Subindustry:** {r['arkap_subindustry']}")
                        if 'nace_conversion_status' in r:
                            st.caption(f"Match: {r['nace_conversion_status']}")

                    st.subheader("💰 Financial Data")
                    c1,c2,c3=st.columns(3)
                    with c1:
                        if 'last_yr' in r: st.metric("Year",r['last_yr'])
                        if 'employees' in r: st.metric("Emp",safe_format(r.get('employees')))
                    with c2:
                        if 'value_of_production_th' in r:
                            st.metric("Prod",safe_format(r.get('value_of_production_th'),pre="€",suf="k"))
                        if 'ebitda_th' in r:
                            st.metric("EBITDA",safe_format(r.get('ebitda_th'),pre="€",suf="k"))
                    with c3:
                        if 'pfn_th' in r:
                            st.metric("PFN",safe_format(r.get('pfn_th'),pre="€",suf="k"))

                else:
                    st.subheader("🌐 Web Extraction Results")
                    extracted_data = {k: v for k, v in r.items()
                        if k not in ['company_name','website','status','source','search_method','country_code']}

                    if extracted_data:
                        for key, value in extracted_data.items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.info("✓ Company verified but no additional codes extracted from website")

            else:
                st.warning("❌ Not found in database or website")

            with st.expander("🔍 Raw Data"):
                st.json(r)

def main():
    st.set_page_config(page_title="arKap Platform", page_icon="⚡", layout="wide")
    auth = AuthenticationManager()

    if auth.is_valid():
        show_main()
    else:
        show_auth(auth)

if __name__ == "__main__": 
    main()
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
