"""
Custom Risk Categories Management for LLM Risk Visualizer
Allows users to define, manage, and configure custom risk categories
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
from database import DatabaseManager

@dataclass
class RiskCategory:
    """Custom risk category definition"""
    id: str
    name: str
    description: str
    severity_weight: float  # 0.0 to 1.0
    color: str
    icon: str
    threshold_low: float
    threshold_medium: float
    threshold_high: float
    is_active: bool = True
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""

class RiskCategoryManager:
    """Manage custom risk categories"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.init_categories_table()
        self.load_default_categories()
    
    def init_categories_table(self):
        """Initialize risk categories table"""
        query = '''
            CREATE TABLE IF NOT EXISTS risk_categories (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                severity_weight REAL DEFAULT 0.5,
                color TEXT DEFAULT '#ff7f0e',
                icon TEXT DEFAULT '⚠️',
                threshold_low REAL DEFAULT 0.1,
                threshold_medium REAL DEFAULT 0.3,
                threshold_high REAL DEFAULT 0.5,
                is_active BOOLEAN DEFAULT 1,
                created_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        '''
        self.db.connection.execute_update(query)
    
    def load_default_categories(self):
        """Load default risk categories if none exist"""
        existing = self.get_all_categories()
        
        if not existing:
            default_categories = [
                RiskCategory(
                    id="hallucination",
                    name="Hallucination",
                    description="Generation of false or fabricated information",
                    severity_weight=0.8,
                    color="#d62728",
                    icon="🔮",
                    threshold_low=0.1,
                    threshold_medium=0.3,
                    threshold_high=0.5,
                    created_by="system"
                ),
                RiskCategory(
                    id="refusal",
                    name="Refusal",
                    description="Inappropriate refusal to answer legitimate queries",
                    severity_weight=0.4,
                    color="#ff7f0e",
                    icon="🚫",
                    threshold_low=0.05,
                    threshold_medium=0.15,
                    threshold_high=0.3,
                    created_by="system"
                ),
                RiskCategory(
                    id="bias",
                    name="Bias",
                    description="Unfair or prejudiced responses",
                    severity_weight=0.9,
                    color="#8c564b",
                    icon="⚖️",
                    threshold_low=0.1,
                    threshold_medium=0.25,
                    threshold_high=0.4,
                    created_by="system"
                ),
                RiskCategory(
                    id="toxicity",
                    name="Toxicity",
                    description="Harmful, offensive, or inappropriate content",
                    severity_weight=1.0,
                    color="#e377c2",
                    icon="☠️",
                    threshold_low=0.05,
                    threshold_medium=0.15,
                    threshold_high=0.3,
                    created_by="system"
                ),
                RiskCategory(
                    id="privacy_leakage",
                    name="Privacy Leakage",
                    description="Unauthorized disclosure of personal information",
                    severity_weight=0.9,
                    color="#7f7f7f",
                    icon="🔓",
                    threshold_low=0.02,
                    threshold_medium=0.1,
                    threshold_high=0.2,
                    created_by="system"
                ),
                RiskCategory(
                    id="factual_error",
                    name="Factual Error",
                    description="Incorrect factual information or misinformation",
                    severity_weight=0.7,
                    color="#bcbd22",
                    icon="❌",
                    threshold_low=0.1,
                    threshold_medium=0.3,
                    threshold_high=0.5,
                    created_by="system"
                )
            ]
            
            for category in default_categories:
                self.add_category(category)
    
    def add_category(self, category: RiskCategory) -> bool:
        """Add a new risk category"""
        try:
            query = '''
                INSERT INTO risk_categories 
                (id, name, description, severity_weight, color, icon, 
                 threshold_low, threshold_medium, threshold_high, is_active, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                category.id, category.name, category.description,
                category.severity_weight, category.color, category.icon,
                category.threshold_low, category.threshold_medium, category.threshold_high,
                category.is_active, category.created_by
            )
            
            return self.db.connection.execute_update(query, params)
        except Exception as e:
            print(f"Error adding category: {e}")
            return False
    
    def update_category(self, category: RiskCategory) -> bool:
        """Update an existing risk category"""
        try:
            query = '''
                UPDATE risk_categories 
                SET name=?, description=?, severity_weight=?, color=?, icon=?,
                    threshold_low=?, threshold_medium=?, threshold_high=?, 
                    is_active=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
            '''
            
            params = (
                category.name, category.description, category.severity_weight,
                category.color, category.icon, category.threshold_low,
                category.threshold_medium, category.threshold_high,
                category.is_active, category.id
            )
            
            return self.db.connection.execute_update(query, params)
        except Exception as e:
            print(f"Error updating category: {e}")
            return False
    
    def delete_category(self, category_id: str) -> bool:
        """Delete a risk category (soft delete by deactivating)"""
        try:
            query = '''
                UPDATE risk_categories 
                SET is_active=0, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
            '''
            return self.db.connection.execute_update(query, (category_id,))
        except Exception as e:
            print(f"Error deleting category: {e}")
            return False
    
    def get_category(self, category_id: str) -> Optional[RiskCategory]:
        """Get a specific risk category"""
        query = "SELECT * FROM risk_categories WHERE id=?"
        result = self.db.connection.execute_query(query, (category_id,))
        
        if not result.empty:
            row = result.iloc[0]
            return RiskCategory(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                severity_weight=row['severity_weight'],
                color=row['color'],
                icon=row['icon'],
                threshold_low=row['threshold_low'],
                threshold_medium=row['threshold_medium'],
                threshold_high=row['threshold_high'],
                is_active=bool(row['is_active']),
                created_by=row.get('created_by', ''),
                created_at=row.get('created_at', ''),
                updated_at=row.get('updated_at', '')
            )
        return None
    
    def get_all_categories(self, active_only: bool = True) -> List[RiskCategory]:
        """Get all risk categories"""
        query = "SELECT * FROM risk_categories"
        if active_only:
            query += " WHERE is_active=1"
        query += " ORDER BY name"
        
        result = self.db.connection.execute_query(query)
        categories = []
        
        for _, row in result.iterrows():
            categories.append(RiskCategory(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                severity_weight=row['severity_weight'],
                color=row['color'],
                icon=row['icon'],
                threshold_low=row['threshold_low'],
                threshold_medium=row['threshold_medium'],
                threshold_high=row['threshold_high'],
                is_active=bool(row['is_active']),
                created_by=row.get('created_by', ''),
                created_at=row.get('created_at', ''),
                updated_at=row.get('updated_at', '')
            ))
        
        return categories
    
    def get_categories_dict(self, active_only: bool = True) -> Dict[str, RiskCategory]:
        """Get categories as a dictionary"""
        categories = self.get_all_categories(active_only)
        return {cat.id: cat for cat in categories}
    
    def calculate_weighted_risk_score(self, risk_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate weighted risk scores using custom categories"""
        categories = self.get_categories_dict()
        
        # Add weights to risk data
        risk_data['severity_weight'] = risk_data['Risk_Category'].map(
            lambda cat: categories.get(cat.lower().replace(' ', '_'), 
                                     RiskCategory('unknown', cat, '', 0.5, '', '', 0, 0, 0)).severity_weight
        )
        
        # Calculate weighted risk score
        risk_data['weighted_risk_score'] = risk_data['Risk_Rate'] * risk_data['severity_weight']
        
        return risk_data
    
    def export_categories(self, file_path: str) -> bool:
        """Export categories to JSON file"""
        try:
            categories = self.get_all_categories(active_only=False)
            categories_dict = [asdict(cat) for cat in categories]
            
            with open(file_path, 'w') as f:
                json.dump(categories_dict, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting categories: {e}")
            return False
    
    def import_categories(self, file_path: str) -> Tuple[bool, str]:
        """Import categories from JSON file"""
        try:
            with open(file_path, 'r') as f:
                categories_data = json.load(f)
            
            imported_count = 0
            for cat_data in categories_data:
                category = RiskCategory(**cat_data)
                if self.add_category(category):
                    imported_count += 1
            
            return True, f"Successfully imported {imported_count} categories"
        except Exception as e:
            return False, f"Error importing categories: {e}"

def render_risk_category_management():
    """Render the risk category management interface"""
    
    st.header("🏷️ Risk Category Management")
    
    # Initialize manager
    db_manager = st.session_state.get('db_manager')
    if not db_manager:
        from database import DatabaseManager
        db_manager = DatabaseManager()
        st.session_state.db_manager = db_manager
    
    category_manager = RiskCategoryManager(db_manager)
    
    # Tabs for different operations
    tab1, tab2, tab3, tab4 = st.tabs(["📋 View Categories", "➕ Add Category", "✏️ Edit Category", "📊 Analytics"])
    
    with tab1:
        st.subheader("Current Risk Categories")
        
        categories = category_manager.get_all_categories()
        
        if categories:
            # Display categories in a nice format
            for category in categories:
                with st.expander(f"{category.icon} {category.name}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Description:** {category.description}")
                        st.write(f"**Severity Weight:** {category.severity_weight}")
                        st.write(f"**Active:** {'✅' if category.is_active else '❌'}")
                    
                    with col2:
                        st.write(f"**Color:** <span style='color: {category.color}'>●</span> {category.color}", unsafe_allow_html=True)
                        st.write(f"**Thresholds:**")
                        st.write(f"- Low: {category.threshold_low}")
                        st.write(f"- Medium: {category.threshold_medium}")
                        st.write(f"- High: {category.threshold_high}")
                    
                    with col3:
                        st.write(f"**Created by:** {category.created_by}")
                        st.write(f"**Created:** {category.created_at}")
                        st.write(f"**Updated:** {category.updated_at}")
                        
                        # Quick action buttons
                        col3a, col3b = st.columns(2)
                        with col3a:
                            if st.button(f"Edit", key=f"edit_{category.id}"):
                                st.session_state.edit_category_id = category.id
                                st.rerun()
                        
                        with col3b:
                            if category.created_by != "system":  # Don't allow deleting system categories
                                if st.button(f"Delete", key=f"delete_{category.id}"):
                                    if category_manager.delete_category(category.id):
                                        st.success(f"Category '{category.name}' deleted successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete category")
        else:
            st.info("No risk categories found. Add some categories to get started.")
        
        # Export/Import options
        st.subheader("Import/Export")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Categories"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"risk_categories_export_{timestamp}.json"
                
                if category_manager.export_categories(file_path):
                    st.success(f"Categories exported to {file_path}")
                    
                    # Provide download link
                    with open(file_path, 'r') as f:
                        st.download_button(
                            label="Download Export File",
                            data=f.read(),
                            file_name=file_path,
                            mime="application/json"
                        )
                else:
                    st.error("Failed to export categories")
        
        with col2:
            uploaded_file = st.file_uploader("📤 Import Categories", type=['json'])
            if uploaded_file:
                if st.button("Import"):
                    # Save uploaded file temporarily
                    temp_path = f"temp_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.read())
                    
                    # Import categories
                    success, message = category_manager.import_categories(temp_path)
                    
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    with tab2:
        st.subheader("Add New Risk Category")
        
        with st.form("add_category_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                category_id = st.text_input("Category ID", help="Unique identifier (lowercase, underscore-separated)")
                category_name = st.text_input("Category Name")
                category_description = st.text_area("Description")
                severity_weight = st.slider("Severity Weight", 0.0, 1.0, 0.5, 0.1)
            
            with col2:
                category_color = st.color_picker("Color", "#ff7f0e")
                category_icon = st.text_input("Icon (emoji)", "⚠️")
                
                st.write("**Thresholds:**")
                threshold_low = st.number_input("Low Threshold", 0.0, 1.0, 0.1, 0.01)
                threshold_medium = st.number_input("Medium Threshold", 0.0, 1.0, 0.3, 0.01)
                threshold_high = st.number_input("High Threshold", 0.0, 1.0, 0.5, 0.01)
            
            submitted = st.form_submit_button("Add Category")
            
            if submitted:
                if not category_id or not category_name:
                    st.error("Category ID and Name are required")
                elif threshold_low >= threshold_medium or threshold_medium >= threshold_high:
                    st.error("Thresholds must be in ascending order: Low < Medium < High")
                else:
                    # Get current user
                    from auth import get_current_user
                    current_user = get_current_user()
                    created_by = current_user['username'] if current_user else 'unknown'
                    
                    new_category = RiskCategory(
                        id=category_id,
                        name=category_name,
                        description=category_description,
                        severity_weight=severity_weight,
                        color=category_color,
                        icon=category_icon,
                        threshold_low=threshold_low,
                        threshold_medium=threshold_medium,
                        threshold_high=threshold_high,
                        created_by=created_by
                    )
                    
                    if category_manager.add_category(new_category):
                        st.success(f"Category '{category_name}' added successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to add category. ID might already exist.")
    
    with tab3:
        st.subheader("Edit Risk Category")
        
        # Category selection
        categories = category_manager.get_all_categories()
        category_options = {cat.name: cat.id for cat in categories}
        
        if category_options:
            # Check if edit category is set in session state
            edit_category_id = st.session_state.get('edit_category_id')
            
            if edit_category_id:
                selected_category_name = next((cat.name for cat in categories if cat.id == edit_category_id), None)
            else:
                selected_category_name = None
            
            selected_category_name = st.selectbox(
                "Select Category to Edit", 
                list(category_options.keys()),
                index=list(category_options.keys()).index(selected_category_name) if selected_category_name else 0
            )
            
            selected_category_id = category_options[selected_category_name]
            category = category_manager.get_category(selected_category_id)
            
            if category:
                with st.form("edit_category_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        new_name = st.text_input("Category Name", value=category.name)
                        new_description = st.text_area("Description", value=category.description)
                        new_severity_weight = st.slider("Severity Weight", 0.0, 1.0, category.severity_weight, 0.1)
                        new_is_active = st.checkbox("Active", value=category.is_active)
                    
                    with col2:
                        new_color = st.color_picker("Color", category.color)
                        new_icon = st.text_input("Icon (emoji)", category.icon)
                        
                        st.write("**Thresholds:**")
                        new_threshold_low = st.number_input("Low Threshold", 0.0, 1.0, category.threshold_low, 0.01)
                        new_threshold_medium = st.number_input("Medium Threshold", 0.0, 1.0, category.threshold_medium, 0.01)
                        new_threshold_high = st.number_input("High Threshold", 0.0, 1.0, category.threshold_high, 0.01)
                    
                    submitted = st.form_submit_button("Update Category")
                    
                    if submitted:
                        if not new_name:
                            st.error("Category name is required")
                        elif new_threshold_low >= new_threshold_medium or new_threshold_medium >= new_threshold_high:
                            st.error("Thresholds must be in ascending order: Low < Medium < High")
                        else:
                            updated_category = RiskCategory(
                                id=category.id,
                                name=new_name,
                                description=new_description,
                                severity_weight=new_severity_weight,
                                color=new_color,
                                icon=new_icon,
                                threshold_low=new_threshold_low,
                                threshold_medium=new_threshold_medium,
                                threshold_high=new_threshold_high,
                                is_active=new_is_active,
                                created_by=category.created_by,
                                created_at=category.created_at
                            )
                            
                            if category_manager.update_category(updated_category):
                                st.success(f"Category '{new_name}' updated successfully!")
                                # Clear edit session state
                                if 'edit_category_id' in st.session_state:
                                    del st.session_state.edit_category_id
                                st.rerun()
                            else:
                                st.error("Failed to update category")
        else:
            st.info("No categories available to edit.")
    
    with tab4:
        st.subheader("Risk Category Analytics")
        
        categories = category_manager.get_all_categories()
        
        if categories:
            # Category distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Category Summary**")
                summary_data = []
                
                for category in categories:
                    summary_data.append({
                        'Category': f"{category.icon} {category.name}",
                        'Severity Weight': category.severity_weight,
                        'Active': '✅' if category.is_active else '❌',
                        'Created By': category.created_by
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            
            with col2:
                st.write("**Severity Weight Distribution**")
                
                import plotly.express as px
                
                weights_data = pd.DataFrame({
                    'Category': [cat.name for cat in categories],
                    'Severity Weight': [cat.severity_weight for cat in categories],
                    'Color': [cat.color for cat in categories]
                })
                
                fig = px.bar(
                    weights_data,
                    x='Category',
                    y='Severity Weight',
                    color='Color',
                    color_discrete_map={color: color for color in weights_data['Color']},
                    title="Severity Weights by Category"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Threshold comparison
            st.write("**Threshold Comparison**")
            
            threshold_data = []
            for category in categories:
                threshold_data.extend([
                    {'Category': category.name, 'Threshold': 'Low', 'Value': category.threshold_low},
                    {'Category': category.name, 'Threshold': 'Medium', 'Value': category.threshold_medium},
                    {'Category': category.name, 'Threshold': 'High', 'Value': category.threshold_high}
                ])
            
            threshold_df = pd.DataFrame(threshold_data)
            
            fig_thresholds = px.bar(
                threshold_df,
                x='Category',
                y='Value',
                color='Threshold',
                barmode='group',
                title="Risk Thresholds by Category"
            )
            
            st.plotly_chart(fig_thresholds, use_container_width=True)
        else:
            st.info("No categories available for analytics.")

# Integration with main app
def integrate_custom_categories():
    """Integration function to add custom categories to the main app"""
    
    # This function can be called from the main app to add custom category management
    # to the admin panel or as a separate page
    
    render_risk_category_management()

if __name__ == "__main__":
    # For testing purposes
    render_risk_category_management()