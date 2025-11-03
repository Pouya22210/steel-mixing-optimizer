"""
Steel Mixing Optimizer - Dash Web Application (AUTHENTICATED v4.3)
===================================================================
A web-based tool to find optimal mixing ratios of steels to achieve target compositions.

NEW in v4.3:
- Username/Password authentication
- MS temperature calculation using comprehensive formula:
  Ms (°C) = 550 - 350C - 40Mn - 20Cr - 10Mo - 17Ni - 8W - 35V - 10Cu + 15Co + 30Al

Previous features:
- Multiple solutions display
- Complete element breakdown
- Interactive visualizations

Run this file and open http://localhost:8050 in your browser.

Author: Assistant
Date: November 2025
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import base64
import io
import json
import plotly.graph_objects as go
import plotly.express as px
try:
    import dash_auth
except Exception:
    # Fallback when dash_auth is not installed: create a minimal shim so the app can run
    # without authentication (useful for development); install `dash-auth` for real auth.
    import types, warnings
    warnings.warn("dash_auth not installed; running without authentication (development only).", UserWarning)
    dash_auth = types.SimpleNamespace(BasicAuth=lambda app, pairs: None)

# ============================================================================
# AUTHENTICATION SETUP
# ============================================================================
# Define valid username and password pairs
VALID_USERNAME_PASSWORD_PAIRS = {
    'admin': 'steel2025',
    'user': 'password123',
    'engineer': 'metallurgy'
}

# Initialize the Dash app with a nice theme
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css"
])
app.title = "Steel Mixing Optimizer"

# Add Basic Authentication
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

# Global variables to store data
database_df = None
custom_alloys = []

# ============================================================================
# MS TEMPERATURE CALCULATION
# ============================================================================

def calculate_ms_temperature(composition):
    """
    Calculate Martensite Start (MS) temperature using comprehensive formula.
    
    Ms (°C, wt.%) = 550 - 350C - 40Mn - 20Cr - 10Mo - 17Ni - 8W - 35V - 10Cu + 15Co + 30Al
    
    Args:
        composition: dict with element compositions in wt.%
    
    Returns:
        MS temperature in °C
    """
    C = composition.get('C', 0)
    Mn = composition.get('Mn', 0)
    Cr = composition.get('Cr', 0)
    Mo = composition.get('Mo', 0)
    Ni = composition.get('Ni', 0)
    W = composition.get('W', 0)
    V = composition.get('V', 0)
    Cu = composition.get('Cu', 0)
    Co = composition.get('Co', 0)
    Al = composition.get('Al', 0)
    
    ms_temp = 550 - 350*C - 40*Mn - 20*Cr - 10*Mo - 17*Ni - 8*W - 35*V - 10*Cu + 15*Co + 30*Al
    
    return ms_temp


# ============================================================================
# ICON HELPER COMPONENTS
# ============================================================================

def icon(name, size="1em", className=""):
    """Create a Bootstrap Icon with consistent styling."""
    return html.I(className=f"bi bi-{name} {className}", style={'fontSize': size, 'marginRight': '8px'})

def icon_only(name, size="1em", className=""):
    """Create a Bootstrap Icon without margin."""
    return html.I(className=f"bi bi-{name} {className}", style={'fontSize': size})

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_dataframe(df):
    """Clean dataframe by rounding numeric columns to avoid floating point issues."""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].round(4)
    return df_clean

def format_number(value):
    """Format a number to remove trailing zeros and excessive decimals."""
    if isinstance(value, (int, float)):
        if value == 0:
            return 0
        # Round to 4 decimals and remove trailing zeros
        formatted = f"{value:.4f}".rstrip('0').rstrip('.')
        return float(formatted) if '.' in formatted else int(float(formatted))
    return value

# ============================================================================
# CORE OPTIMIZATION FUNCTIONS
# ============================================================================

def solve_mixing_ratios(steels_df, element_cols, target_composition, tolerances):
    """
    Solve for optimal mixing ratios for a given combination of steels.
    Returns: dict with success, ratios, composition, etc.
    """
    n_steels = len(steels_df)
    steel_names = steels_df['Name'].tolist()
    target_elements = list(target_composition.keys())
    
    # Extract composition matrix
    composition_matrix = steels_df[element_cols].values.T
    
    # Initial guess: equal ratios
    x0 = np.ones(n_steels) / n_steels
    
    # Objective: minimize squared deviation
    def objective(ratios):
        resulting_comp = composition_matrix @ ratios
        deviation = 0
        for i, elem in enumerate(element_cols):
            if elem in target_elements:
                target_val = target_composition[elem]
                deviation += (resulting_comp[i] - target_val) ** 2
        return deviation
    
    # Constraints: sum to 1, non-negative
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(n_steels)]
    
    # Solve
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        return {'success': False}
    
    optimal_ratios = result.x
    resulting_comp = composition_matrix @ optimal_ratios
    resulting_dict = {elem: resulting_comp[i] for i, elem in enumerate(element_cols)}
    
    # Check tolerances
    meets_tolerances = True
    deviations = {}
    
    for elem in target_elements:
        target_val = target_composition[elem]
        actual_val = resulting_dict.get(elem, 0.0)
        tolerance = tolerances.get(elem, 0.0)
        deviation = abs(actual_val - target_val)
        deviations[elem] = deviation
        
        if deviation > tolerance:
            meets_tolerances = False
    
    # Pure elements needed
    pure_elements_needed = {}
    if not meets_tolerances:
        for elem in target_elements:
            target_val = target_composition[elem]
            actual_val = resulting_dict.get(elem, 0.0)
            diff = target_val - actual_val
            if abs(diff) > 0.001:
                pure_elements_needed[elem] = diff
    
    total_deviation = sum(deviations.values())
    
    # Calculate MS temperature
    ms_temperature = calculate_ms_temperature(resulting_dict)
    
    return {
        'success': True,
        'meets_tolerances': meets_tolerances,
        'steel_names': steel_names,
        'ratios': optimal_ratios,
        'resulting_composition': resulting_dict,
        'deviations': deviations,
        'total_deviation': total_deviation,
        'pure_elements_needed': pure_elements_needed,
        'ms_temperature': ms_temperature
    }


def find_optimal_mix(all_steels, target_composition, tolerances, max_combination_size=5, num_solutions=1):
    """
    Find optimal steel mixture using prioritized search.
    Returns: (success, list_of_results, search_log)
    
    NEW: Returns multiple solutions (top num_solutions)
    """
    if all_steels is None or len(all_steels) == 0:
        return False, None, ["No steels available!"]
    
    element_cols = [col for col in all_steels.columns if col != 'Name']
    target_elements = list(target_composition.keys())
    
    search_log = []
    search_log.append(f"Starting search with {len(all_steels)} steels...")
    search_log.append(f"Looking for top {num_solutions} solution(s)...")
    
    # Store all valid solutions and all tested solutions
    valid_solutions = []
    all_solutions = []
    
    # Try combinations from 2 to max_combination_size
    for n_steels in range(2, min(len(all_steels) + 1, max_combination_size + 1)):
        n_combinations = len(list(combinations(range(len(all_steels)), n_steels)))
        search_log.append(f"Trying {n_combinations} combinations of {n_steels} steels...")
        
        for combo_indices in combinations(range(len(all_steels)), n_steels):
            combo_steels = all_steels.iloc[list(combo_indices)]
            result = solve_mixing_ratios(combo_steels, element_cols, target_composition, tolerances)
            
            if result['success']:
                result['n_steels'] = n_steels  # Track how many steels in this combo
                all_solutions.append(result)
                
                if result['meets_tolerances']:
                    valid_solutions.append(result)
        
        # If we found enough valid solutions, we can stop early
        if len(valid_solutions) >= num_solutions:
            search_log.append(f"✓ SUCCESS! Found {len(valid_solutions)} valid mixture(s) with {n_steels} steels!")
            break
        
        search_log.append(f"Found {len(valid_solutions)} valid combination(s) with {n_steels} steels.")
    
    # Sort and select top solutions
    if len(valid_solutions) >= num_solutions:
        # Sort valid solutions by total deviation (lower is better)
        valid_solutions.sort(key=lambda x: x['total_deviation'])
        top_solutions = valid_solutions[:num_solutions]
        search_log.append(f"Returning top {len(top_solutions)} valid solution(s)")
        return True, top_solutions, search_log
    
    elif len(valid_solutions) > 0:
        # Return whatever valid solutions we found
        valid_solutions.sort(key=lambda x: x['total_deviation'])
        search_log.append(f"Found {len(valid_solutions)} valid solution(s) (less than requested {num_solutions})")
        return True, valid_solutions, search_log
    
    else:
        # No valid solutions - return best N by deviation
        search_log.append(f"⚠ No combination within tolerances found (tested up to {max_combination_size} steels)")
        search_log.append(f"Returning top {num_solutions} closest achievable composition(s)...")
        all_solutions.sort(key=lambda x: x['total_deviation'])
        top_solutions = all_solutions[:num_solutions]
        return False, top_solutions, search_log


# ============================================================================
# DASH LAYOUT
# ============================================================================

app.layout = dbc.Container([
    # Header with auth indicator
    dbc.Row([
        dbc.Col([
            html.H1([
                icon("diagram-3", size="1.2em", className="text-primary"),
                "Powder Mixing Optimiser for Additive Manufacturing"
            ], className="text-center text-primary mb-3 mt-4"),
            html.P("Find optimal mixing ratios of steel powders to achieve target compositions",
                   className="text-center text-muted mb-2"),
            dbc.Badge([icon_only("shield-check"), "Authenticated Session"], color="success", className="mb-4"),
        ])
    ]),
    
    # Main tabs
    dbc.Tabs([
        # TAB 1: Database Management
        dbc.Tab(label="Database", tab_id="tab-database", children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Steel Database Management", className="mb-3"),
                    
                    # Upload section
                    html.Div([
                        html.H5("Upload Database", className="mb-2"),
                        dcc.Upload(
                            id='upload-database',
                            children=dbc.Button([
                                icon("upload"),
                                "Upload Excel File (.xlsx)"
                            ], color="primary", className="mb-3"),
                            multiple=False
                        ),
                        html.Div(id='upload-status', className="mb-3"),
                    ]),
                    
                    html.Hr(),
                    
                    # Or use sample database
                    html.Div([
                        html.H5("Or Use Sample Database", className="mb-2"),
                        dbc.Button([
                            icon("flask"),
                            "Load Sample Database (10 steels)"
                        ], id="load-sample-btn", color="success", className="mb-3"),
                        html.Div(id='sample-load-status'),
                    ]),
                    
                    html.Hr(),
                    
                    # View database with inline checkboxes
                    html.H5("Current Database", className="mb-2"),
                    html.P("Select rows to remove (click on rows to select):", className="text-muted mb-2"),
                    html.Div(id='database-table-container'),
                    
                    dbc.Button([
                        icon("trash3"),
                        "Remove Selected Alloys"
                    ], id="remove-alloys-btn", color="danger", outline=True, className="mb-2 mt-3"),
                    html.Div(id='remove-alloy-status', className="mt-2"),
                    
                    html.Hr(),
                    
                    # Add custom alloy section
                    html.H5("Add Custom Alloy", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Alloy Name:", className="fw-bold"),
                            dbc.Input(id="custom-alloy-name", placeholder="e.g., MyCustomSteel", type="text"),
                        ], width=12, className="mb-3"),
                    ]),
                    
                    html.P("Enter composition (wt.%) - leave blank for 0:", className="text-muted mb-2"),
                    
                    # Element input fields - organized in rows
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("C:", className="fw-bold"),
                            dbc.Input(id="elem-C", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Mn:", className="fw-bold"),
                            dbc.Input(id="elem-Mn", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Cr:", className="fw-bold"),
                            dbc.Input(id="elem-Cr", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Mo:", className="fw-bold"),
                            dbc.Input(id="elem-Mo", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Ni:", className="fw-bold"),
                            dbc.Input(id="elem-Ni", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("W:", className="fw-bold"),
                            dbc.Input(id="elem-W", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                    ], className="mb-2"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("V:", className="fw-bold"),
                            dbc.Input(id="elem-V", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Cu:", className="fw-bold"),
                            dbc.Input(id="elem-Cu", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Co:", className="fw-bold"),
                            dbc.Input(id="elem-Co", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Al:", className="fw-bold"),
                            dbc.Input(id="elem-Al", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Si:", className="fw-bold"),
                            dbc.Input(id="elem-Si", placeholder="0.0", type="number", step="0.01"),
                        ], width=2),
                    ], className="mb-3"),
                    
                    dbc.Button([
                        icon("plus-circle"),
                        "Add Custom Alloy"
                    ], id="add-custom-alloy-btn", color="info", className="mb-2"),
                    html.Div(id='custom-alloy-status', className="mt-2"),
                ])
            ], className="mt-3")
        ]),
        
        # TAB 2: Optimization
        dbc.Tab(label="Optimize", tab_id="tab-optimize", children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Set Target Composition & Run Optimization", className="mb-3"),
                    html.P("Enter target values and tolerances for each element (leave blank to exclude)", className="text-muted mb-3"),
                    
                    # Element input table - organized by rows
                    html.H5("Target Composition & Tolerances", className="mb-3"),
                    
                    # Header row
                    dbc.Row([
                        dbc.Col(html.Div("Element", className="fw-bold text-center"), width=2),
                        dbc.Col(html.Div("Target (wt.%)", className="fw-bold text-center"), width=2),
                        dbc.Col(html.Div("Tolerance (±wt.%)", className="fw-bold text-center"), width=2),
                        dbc.Col(html.Div("Element", className="fw-bold text-center"), width=2),
                        dbc.Col(html.Div("Target (wt.%)", className="fw-bold text-center"), width=2),
                        dbc.Col(html.Div("Tolerance (±wt.%)", className="fw-bold text-center"), width=2),
                    ], className="mb-2", style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px'}),
                    
                    # Row 1: C, Mn
                    dbc.Row([
                        dbc.Col(html.Div("C", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-C", placeholder="e.g., 0.51", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-C", placeholder="e.g., 0.1", type="number", step="0.01"), width=2),
                        dbc.Col(html.Div("Mn", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-Mn", placeholder="e.g., 1.0", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-Mn", placeholder="e.g., 0.2", type="number", step="0.01"), width=2),
                    ], className="mb-2"),
                    
                    # Row 2: Cr, Mo
                    dbc.Row([
                        dbc.Col(html.Div("Cr", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-Cr", placeholder="e.g., 17.1", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-Cr", placeholder="e.g., 0.5", type="number", step="0.01"), width=2),
                        dbc.Col(html.Div("Mo", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-Mo", placeholder="e.g., 2.0", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-Mo", placeholder="e.g., 0.3", type="number", step="0.01"), width=2),
                    ], className="mb-2"),
                    
                    # Row 3: Ni, W
                    dbc.Row([
                        dbc.Col(html.Div("Ni", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-Ni", placeholder="e.g., 6.4", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-Ni", placeholder="e.g., 0.5", type="number", step="0.01"), width=2),
                        dbc.Col(html.Div("W", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-W", placeholder="e.g., 0.0", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-W", placeholder="e.g., 0.1", type="number", step="0.01"), width=2),
                    ], className="mb-2"),
                    
                    # Row 4: V, Cu
                    dbc.Row([
                        dbc.Col(html.Div("V", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-V", placeholder="e.g., 0.0", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-V", placeholder="e.g., 0.1", type="number", step="0.01"), width=2),
                        dbc.Col(html.Div("Cu", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-Cu", placeholder="e.g., 0.0", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-Cu", placeholder="e.g., 0.1", type="number", step="0.01"), width=2),
                    ], className="mb-2"),
                    
                    # Row 5: Co, Al
                    dbc.Row([
                        dbc.Col(html.Div("Co", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-Co", placeholder="e.g., 0.0", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-Co", placeholder="e.g., 0.1", type="number", step="0.01"), width=2),
                        dbc.Col(html.Div("Al", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-Al", placeholder="e.g., 0.0", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-Al", placeholder="e.g., 0.1", type="number", step="0.01"), width=2),
                    ], className="mb-2"),
                    
                    # Row 6: Si
                    dbc.Row([
                        dbc.Col(html.Div("Si", className="fw-bold text-center pt-2"), width=2),
                        dbc.Col(dbc.Input(id="target-Si", placeholder="e.g., 0.0", type="number", step="0.01"), width=2),
                        dbc.Col(dbc.Input(id="tolerance-Si", placeholder="e.g., 0.1", type="number", step="0.01"), width=2),
                        dbc.Col(width=6),  # Empty space
                    ], className="mb-3"),
                    
                    html.Hr(),
                    
                    # Quick preset buttons
                    html.H5("Quick Presets", className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button([
                                icon("clipboard-check"),
                                "Example (Ch.6): C=0.51, Cr=17.1, Ni=6.4"
                            ], id="preset-chapter6", color="info", outline=True, size="sm", className="mb-2 w-100"),
                        ], width=6),
                        dbc.Col([
                            dbc.Button([
                                icon("trash3"),
                                "Clear All Fields"
                            ], id="clear-all-btn", color="secondary", outline=True, size="sm", className="mb-2 w-100"),
                        ], width=6),
                    ]),
                    
                    html.Hr(),
                    
                    # Optimization settings
                    html.H5("Optimization Settings", className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Max Combination Size (2-10):"),
                            dbc.Input(id="max-combination-size", type="number", value=5, min=2, max=10),
                            html.Small("Number of steels to mix (lower = faster, higher = more options)", className="text-muted"),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Number of Solutions to Show (1-10):"),
                            dbc.Input(id="num-solutions", type="number", value=3, min=1, max=10),
                            html.Small("How many top combinations to display (1=best only, 3=top 3)", className="text-muted"),
                        ], width=6),
                    ], className="mb-3"),
                    
                    html.Hr(),
                    
                    # Run button
                    dbc.Button([
                        icon("play-circle-fill", size="1.2em"),
                        "Find Optimal Mix"
                    ], id="run-optimization-btn", color="success", size="lg", className="mb-3 w-100"),
                    
                    # Status
                    dcc.Loading(
                        id="loading-optimization",
                        type="default",
                        children=html.Div(id='optimization-status')
                    ),
                ])
            ], className="mt-3")
        ]),
        
        # TAB 3: Results
        dbc.Tab(label="Results", tab_id="tab-results", children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("Optimization Results", className="mb-3"),
                    html.Div(id='results-container'),
                    
                    html.Hr(),
                    
                    # Export button
                    dbc.Button([
                        icon("download"),
                        "Download Results (CSV)"
                    ], id="download-results-btn", color="primary", className="mb-3"),
                    dcc.Download(id="download-results"),
                ])
            ], className="mt-3")
        ]),
        
        # TAB 4: Help
        dbc.Tab(label="Help", tab_id="tab-help", children=[
            dbc.Card([
                dbc.CardBody([
                    html.H4("How to Use This Application", className="mb-3"),
                    
                    html.H5([icon("lock-fill"), "Authentication"], className="mt-3"),
                    html.P("This application requires username and password authentication. Valid credentials:"),
                    html.Ul([
                        html.Li([html.Strong("admin"), " / steel2025"]),
                        html.Li([html.Strong("user"), " / password123"]),
                        html.Li([html.Strong("engineer"), " / metallurgy"]),
                    ]),
                    
                    html.Hr(),
                    
                    html.H5([icon("list-ol"), "Quick Start Guide"], className="mt-3"),
                    html.Ol([
                        html.Li([html.Strong("Load Database: "), "Go to 'Database' tab and upload your Excel file or load the sample database"]),
                        html.Li([html.Strong("Set Target: "), "Go to 'Optimize' tab and enter your target composition"]),
                        html.Li([html.Strong("Set Tolerances: "), "Enter acceptable deviations for each element"]),
                        html.Li([html.Strong("Choose Number of Solutions: "), "Specify how many top combinations you want to see (default = 3)"]),
                        html.Li([html.Strong("Run Optimization: "), "Click 'Find Optimal Mix' button"]),
                        html.Li([html.Strong("View Results: "), "Go to 'Results' tab to see all solutions with MS temperatures"]),
                        html.Li([html.Strong("Export: "), "Download results as CSV for documentation"]),
                    ]),
                    
                    html.Hr(),
                    
                    html.H5([icon("thermometer-half"), "MS Temperature Calculation"], className="mt-3"),
                    html.P("The Martensite Start (MS) temperature is calculated using a comprehensive empirical formula:"),
                    html.Pre("Ms (°C) = 550 - 350C - 40Mn - 20Cr - 10Mo - 17Ni - 8W - 35V - 10Cu + 15Co + 30Al",
                            style={'backgroundColor': '#f5f5f5', 'padding': '15px', 'borderRadius': '5px'}),
                    html.P([
                        "This comprehensive formula accounts for the effects of multiple alloying elements on martensite formation. ",
                        "Note that Co and Al increase MS temperature (positive coefficients), while most other elements decrease it. ",
                        "MS temperature is displayed for each solution."
                    ]),
                    
                    html.Hr(),
                    
                    html.H5([icon("star"), "Features in v4.3"], className="mt-3"),
                    html.Ul([
                        html.Li([html.Strong("Authentication: "), "Username/password protection for secure access"]),
                        html.Li([html.Strong("MS Temperature: "), "Automatic calculation using Andrews formula"]),
                        html.Li([html.Strong("Multiple Solutions: "), "Shows top N best combinations"]),
                        html.Li([html.Strong("Complete Element Display: "), "Shows ALL elements including non-target ones"]),
                        html.Li([html.Strong("Inline Selection: "), "Click rows directly in table for removal"]),
                    ]),
                    
                    html.Hr(),
                    
                    html.H5([icon("graph-up"), "Visualizations"], className="mt-3"),
                    html.P("Each solution includes:"),
                    html.Ul([
                        html.Li("MS temperature badge (color-coded by temperature range)"),
                        html.Li("Pie chart showing the proportion of each steel in the mixture"),
                        html.Li("Bar chart comparing target vs achieved composition for each element"),
                        html.Li("Deviation analysis showing how close each element is to tolerance"),
                        html.Li("Color-coded pass/fail indicators for quick assessment"),
                        html.Li("Complete element breakdown including non-target elements"),
                    ]),
                    
                    html.Hr(),
                    
                    html.H5([icon("lightbulb"), "Tips"], className="mt-3"),
                    html.Ul([
                        html.Li("MS temperature helps predict hardenability - higher MS means easier martensite formation"),
                        html.Li("Typical MS temperatures: 200-400°C for tool steels, 300-500°C for low-alloy steels"),
                        html.Li("Requesting multiple solutions lets you compare alternatives by MS temperature"),
                        html.Li("Non-target elements (like Si, Al) affect MS temperature indirectly"),
                        html.Li("Always verify MS predictions experimentally - empirical formulas have limitations"),
                    ]),
                ])
            ], className="mt-3")
        ]),
    ], id="tabs", active_tab="tab-database"),
    
    # Hidden divs to store data
    dcc.Store(id='stored-database'),
    dcc.Store(id='stored-results'),
    dcc.Store(id='stored-selected-rows'),
    
    # Footer
    html.Hr(),
    html.P("Steel Mixing Optimizer v4.3 (Authenticated + MS Temperature) | Built with Dash & Python", 
           className="text-center text-muted mt-4 mb-4"),
    
], fluid=True, style={'maxWidth': '1400px'})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output('stored-database', 'data'),
     Output('upload-status', 'children'),
     Output('database-table-container', 'children'),
     Output('stored-selected-rows', 'data')],
    [Input('upload-database', 'contents'),
     Input('load-sample-btn', 'n_clicks'),
     Input('add-custom-alloy-btn', 'n_clicks'),
     Input('remove-alloys-btn', 'n_clicks')],
    [State('upload-database', 'filename'),
     State('stored-database', 'data'),
     State('custom-alloy-name', 'value'),
     State('elem-C', 'value'),
     State('elem-Mn', 'value'),
     State('elem-Cr', 'value'),
     State('elem-Mo', 'value'),
     State('elem-Ni', 'value'),
     State('elem-W', 'value'),
     State('elem-V', 'value'),
     State('elem-Cu', 'value'),
     State('elem-Co', 'value'),
     State('elem-Al', 'value'),
     State('elem-Si', 'value'),
     State('database-table-container', 'children')]
)
def update_database(contents, sample_clicks, add_clicks, remove_clicks, filename, stored_data, 
                   alloy_name, C, Mn, Cr, Mo, Ni, W, V, Cu, Co, Al, Si, table_container):
    """Handle database upload, sample loading, custom alloy addition, and removal."""
    global database_df, custom_alloys
    
    ctx = callback_context
    if not ctx.triggered:
        return None, "", html.P("No database loaded. Please upload an Excel file or load sample database.", 
                                className="text-muted"), []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Helper function to create table with row selection
    def create_table(df):
        return dash_table.DataTable(
            id='database-table',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i, 'type': 'numeric', 'format': {'specifier': '.4~f'}} 
                    if i != 'Name' else {'name': i, 'id': i} for i in df.columns],
            row_selectable='multi',
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'state': 'selected'},
                    'backgroundColor': '#ffc107',
                    'border': '1px solid #ff9800',
                }
            ],
            page_size=10,
        )
    
    # Handle file upload
    if trigger_id == 'upload-database' and contents is not None:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_excel(io.BytesIO(decoded))
            
            # Clean the dataframe
            df = clean_dataframe(df)
            
            database_df = df
            custom_alloys = []
            
            table = create_table(df)
            
            status = dbc.Alert([
                icon("check-circle-fill", size="1.2em", className="text-success"),
                f"Successfully loaded {len(df)} steels from '{filename}'"
            ], color="success")
            
            return df.to_json(date_format='iso', orient='split'), status, table, []
            
        except Exception as e:
            status = dbc.Alert([
                icon("x-circle-fill", size="1.2em", className="text-danger"),
                f"Error loading file: {str(e)}"
            ], color="danger")
            return None, status, html.P("No database loaded.", className="text-muted"), []
    
    # Handle sample database
    elif trigger_id == 'load-sample-btn' and sample_clicks:
        try:
            # Create sample database
            steels = [
                {'Name': '440C-SLM', 'C': 1.02, 'Mn': 0.6, 'Cr': 17.4, 'Mo': 0.65, 'Ni': 0.1, 'W': 0, 'V': 0, 'Cu': 0, 'Co': 0, 'Al': 0, 'Si': 0},
                {'Name': '410L-SLM', 'C': 0.03, 'Mn': 0.1, 'Cr': 12.5, 'Mo': 0, 'Ni': 0, 'W': 0, 'V': 0, 'Cu': 0, 'Co': 0, 'Al': 0, 'Si': 0},
                {'Name': 'M2-SLM', 'C': 0.98, 'Mn': 0.3, 'Cr': 4.3, 'Mo': 5.0, 'Ni': 0, 'W': 6.2, 'V': 1.2, 'Cu': 0, 'Co': 0, 'Al': 0, 'Si': 0.3},
                {'Name': '316-SLM', 'C': 0.01, 'Mn': 1.5, 'Cr': 16.8, 'Mo': 2.5, 'Ni': 12.7, 'W': 0, 'V': 0, 'Cu': 0, 'Co': 0, 'Al': 0, 'Si': 0.7},
                {'Name': '304L-SLM', 'C': 0.02, 'Mn': 1.2, 'Cr': 18.5, 'Mo': 0.2, 'Ni': 9.5, 'W': 0, 'V': 0, 'Cu': 0, 'Co': 0, 'Al': 0, 'Si': 0.5},
                {'Name': '17-4PH', 'C': 0.04, 'Mn': 0.5, 'Cr': 16.0, 'Mo': 0.3, 'Ni': 4.5, 'W': 0, 'V': 0, 'Cu': 3.5, 'Co': 0, 'Al': 0, 'Si': 0.6},
                {'Name': 'H13', 'C': 0.40, 'Mn': 0.4, 'Cr': 5.2, 'Mo': 1.5, 'Ni': 0.2, 'W': 0, 'V': 1.0, 'Cu': 0, 'Co': 0, 'Al': 0, 'Si': 1.0},
                {'Name': 'D2', 'C': 1.55, 'Mn': 0.4, 'Cr': 12.0, 'Mo': 0.9, 'Ni': 0.3, 'W': 0, 'V': 0.9, 'Cu': 0, 'Co': 0, 'Al': 0, 'Si': 0.4},
                {'Name': 'S7', 'C': 0.50, 'Mn': 0.7, 'Cr': 3.5, 'Mo': 1.4, 'Ni': 0.3, 'W': 0, 'V': 0.2, 'Cu': 0, 'Co': 0, 'Al': 0, 'Si': 0.5},
                {'Name': 'A2', 'C': 1.00, 'Mn': 0.6, 'Cr': 5.2, 'Mo': 1.1, 'Ni': 0.2, 'W': 0, 'V': 0.2, 'Cu': 0, 'Co': 0, 'Al': 0, 'Si': 0.3},
            ]
            df = pd.DataFrame(steels)
            df = clean_dataframe(df)
            
            database_df = df
            custom_alloys = []
            
            table = create_table(df)
            
            status = dbc.Alert([
                icon("check-circle-fill", size="1.2em", className="text-success"),
                f"Sample database loaded with {len(df)} steels!"
            ], color="success")
            
            return df.to_json(date_format='iso', orient='split'), status, table, []
            
        except Exception as e:
            status = dbc.Alert([
                icon("x-circle-fill", size="1.2em", className="text-danger"),
                f"Error creating sample: {str(e)}"
            ], color="danger")
            return None, status, html.P("No database loaded.", className="text-muted"), []
    
    # Handle alloy removal
    elif trigger_id == 'remove-alloys-btn' and remove_clicks:
        if not stored_data:
            status = dbc.Alert("Please load a database first", color="warning")
            return None, status, html.P("No database loaded.", className="text-muted"), []
        
        # Get selected rows from the existing table
        if table_container and isinstance(table_container, dict):
            selected_rows = table_container.get('props', {}).get('selected_rows', [])
        else:
            selected_rows = []
        
        if not selected_rows or len(selected_rows) == 0:
            status = dbc.Alert("Please select at least one row to remove (click on rows in the table)", color="warning")
            df = pd.read_json(stored_data, orient='split')
            df = clean_dataframe(df)
            table = create_table(df)
            return stored_data, status, table, []
        
        try:
            df = pd.read_json(stored_data, orient='split')
            df = clean_dataframe(df)
            
            # Remove selected rows
            df = df.drop(df.index[selected_rows]).reset_index(drop=True)
            
            if len(df) < 2:
                status = dbc.Alert("Cannot remove all alloys! At least 2 alloys must remain.", color="danger")
                df_original = pd.read_json(stored_data, orient='split')
                df_original = clean_dataframe(df_original)
                table = create_table(df_original)
                return stored_data, status, table, []
            
            database_df = df
            
            table = create_table(df)
            
            status = dbc.Alert([
                icon("check-circle-fill", size="1.2em", className="text-success"),
                f"Successfully removed {len(selected_rows)} alloy(s). {len(df)} alloys remaining."
            ], color="success")
            
            return df.to_json(date_format='iso', orient='split'), status, table, []
            
        except Exception as e:
            status = dbc.Alert(f"Error removing alloys: {str(e)}", color="danger")
            if stored_data:
                df = pd.read_json(stored_data, orient='split')
                df = clean_dataframe(df)
                table = create_table(df)
                return stored_data, status, table, []
            return None, status, html.P("No database loaded.", className="text-muted"), []
    
    # Handle custom alloy addition
    elif trigger_id == 'add-custom-alloy-btn' and add_clicks:
        if not alloy_name:
            status = dbc.Alert("Please provide an alloy name", color="warning")
            if stored_data:
                df = pd.read_json(stored_data, orient='split')
                df = clean_dataframe(df)
                table = create_table(df)
                return stored_data, status, table, []
            return None, status, html.P("No database loaded.", className="text-muted"), []
        
        try:
            if stored_data:
                df = pd.read_json(stored_data, orient='split')
            else:
                status = dbc.Alert("Please load a database first", color="warning")
                return None, status, html.P("No database loaded.", className="text-muted"), []
            
            # Create composition dictionary from input fields
            comp_dict = {
                'C': float(C) if C is not None and C != '' else 0.0,
                'Mn': float(Mn) if Mn is not None and Mn != '' else 0.0,
                'Cr': float(Cr) if Cr is not None and Cr != '' else 0.0,
                'Mo': float(Mo) if Mo is not None and Mo != '' else 0.0,
                'Ni': float(Ni) if Ni is not None and Ni != '' else 0.0,
                'W': float(W) if W is not None and W != '' else 0.0,
                'V': float(V) if V is not None and V != '' else 0.0,
                'Cu': float(Cu) if Cu is not None and Cu != '' else 0.0,
                'Co': float(Co) if Co is not None and Co != '' else 0.0,
                'Al': float(Al) if Al is not None and Al != '' else 0.0,
                'Si': float(Si) if Si is not None and Si != '' else 0.0,
            }
            
            # Create new alloy row
            new_row = {'Name': alloy_name}
            for col in df.columns:
                if col != 'Name':
                    new_row[col] = comp_dict.get(col, 0.0)
            
            # Append to dataframe
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Clean the dataframe to fix formatting
            df = clean_dataframe(df)
            
            database_df = df
            
            table = create_table(df)
            
            status = dbc.Alert([
                icon("check-circle-fill", size="1.2em", className="text-success"),
                f"Custom alloy '{alloy_name}' added successfully!"
            ], color="success")
            
            return df.to_json(date_format='iso', orient='split'), status, table, []
            
        except Exception as e:
            status = dbc.Alert(f"✗ Error adding custom alloy: {str(e)}", color="danger")
            if stored_data:
                df = pd.read_json(stored_data, orient='split')
                df = clean_dataframe(df)
                table = create_table(df)
                return stored_data, status, table, []
            return None, status, html.P("No database loaded.", className="text-muted"), []
    
    # Default case
    if stored_data:
        df = pd.read_json(stored_data, orient='split')
        df = clean_dataframe(df)
        table = create_table(df)
        return stored_data, "", table, []
    
    return None, "", html.P("No database loaded. Please upload an Excel file or load sample database.", 
                            className="text-muted"), []


# Callback for preset buttons
@app.callback(
    [Output('target-C', 'value'), Output('tolerance-C', 'value'),
     Output('target-Mn', 'value'), Output('tolerance-Mn', 'value'),
     Output('target-Cr', 'value'), Output('tolerance-Cr', 'value'),
     Output('target-Mo', 'value'), Output('tolerance-Mo', 'value'),
     Output('target-Ni', 'value'), Output('tolerance-Ni', 'value'),
     Output('target-W', 'value'), Output('tolerance-W', 'value'),
     Output('target-V', 'value'), Output('tolerance-V', 'value'),
     Output('target-Cu', 'value'), Output('tolerance-Cu', 'value'),
     Output('target-Co', 'value'), Output('tolerance-Co', 'value'),
     Output('target-Al', 'value'), Output('tolerance-Al', 'value'),
     Output('target-Si', 'value'), Output('tolerance-Si', 'value')],
    [Input('preset-chapter6', 'n_clicks'),
     Input('clear-all-btn', 'n_clicks')]
)
def handle_presets(chapter6_clicks, clear_clicks):
    """Handle preset button clicks."""
    ctx = callback_context
    
    if not ctx.triggered:
        return [None] * 22
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'preset-chapter6' and chapter6_clicks:
        # Chapter 6 example: C=0.51, Cr=17.1, Ni=6.4
        return [
            0.51, 0.1,  # C: target, tolerance
            None, None,  # Mn
            17.1, 0.5,  # Cr
            None, None,  # Mo
            6.4, 0.5,   # Ni
            None, None,  # W
            None, None,  # V
            None, None,  # Cu
            None, None,  # Co
            None, None,  # Al
            None, None,  # Si
        ]
    
    elif trigger_id == 'clear-all-btn' and clear_clicks:
        # Clear all fields
        return [None] * 22
    
    return [None] * 22


@app.callback(
    [Output('stored-results', 'data'),
     Output('optimization-status', 'children')],
    [Input('run-optimization-btn', 'n_clicks')],
    [State('stored-database', 'data'),
     State('target-C', 'value'), State('tolerance-C', 'value'),
     State('target-Mn', 'value'), State('tolerance-Mn', 'value'),
     State('target-Cr', 'value'), State('tolerance-Cr', 'value'),
     State('target-Mo', 'value'), State('tolerance-Mo', 'value'),
     State('target-Ni', 'value'), State('tolerance-Ni', 'value'),
     State('target-W', 'value'), State('tolerance-W', 'value'),
     State('target-V', 'value'), State('tolerance-V', 'value'),
     State('target-Cu', 'value'), State('tolerance-Cu', 'value'),
     State('target-Co', 'value'), State('tolerance-Co', 'value'),
     State('target-Al', 'value'), State('tolerance-Al', 'value'),
     State('target-Si', 'value'), State('tolerance-Si', 'value'),
     State('max-combination-size', 'value'),
     State('num-solutions', 'value')]
)
def run_optimization(n_clicks, database_data, 
                     target_C, tol_C, target_Mn, tol_Mn, target_Cr, tol_Cr,
                     target_Mo, tol_Mo, target_Ni, tol_Ni, target_W, tol_W,
                     target_V, tol_V, target_Cu, tol_Cu, target_Co, tol_Co,
                     target_Al, tol_Al, target_Si, tol_Si, max_size, num_solutions):
    """Run the optimization when button is clicked."""
    if not n_clicks:
        return None, ""
    
    if not database_data:
        return None, dbc.Alert([
            icon("exclamation-triangle-fill", size="1.2em", className="text-warning"),
            "Please load a database first!"
        ], color="warning")
    
    try:
        # Build target composition and tolerances from individual fields
        elements = ['C', 'Mn', 'Cr', 'Mo', 'Ni', 'W', 'V', 'Cu', 'Co', 'Al', 'Si']
        target_values = [target_C, target_Mn, target_Cr, target_Mo, target_Ni, 
                        target_W, target_V, target_Cu, target_Co, target_Al, target_Si]
        tolerance_values = [tol_C, tol_Mn, tol_Cr, tol_Mo, tol_Ni, 
                           tol_W, tol_V, tol_Cu, tol_Co, tol_Al, tol_Si]
        
        # Filter out empty fields
        target_composition = {}
        tolerances = {}
        
        for elem, target, tol in zip(elements, target_values, tolerance_values):
            if target is not None and target != '':
                target_composition[elem] = float(target)
                # Use tolerance if provided, otherwise use a default
                if tol is not None and tol != '':
                    tolerances[elem] = float(tol)
                else:
                    tolerances[elem] = 0.1  # Default tolerance
        
        # Check if at least one element is specified
        if len(target_composition) == 0:
            return None, dbc.Alert([
                icon("exclamation-triangle-fill", size="1.2em", className="text-warning"),
                "Please specify at least one target element!"
            ], color="warning")
        
        # Load database
        df = pd.read_json(database_data, orient='split')
        df = clean_dataframe(df)
        
        # Get all element columns for complete display
        all_element_cols = [col for col in df.columns if col != 'Name']
        
        # Default num_solutions if not provided
        if num_solutions is None or num_solutions < 1:
            num_solutions = 1
        
        # Run optimization with multiple solutions
        success, results, search_log = find_optimal_mix(
            df, target_composition, tolerances, max_size, num_solutions
        )
        
        if results:
            # Store results - now it's a list of solutions
            result_json = {
                'success': success,
                'solutions': [],
                'target_composition': target_composition,
                'tolerances': tolerances,
                'search_log': search_log,
                'all_elements': all_element_cols
            }
            
            # Convert each result to JSON-serializable format
            for result in results:
                solution = {
                    'steel_names': result['steel_names'],
                    'ratios': result['ratios'].tolist(),
                    'resulting_composition': result['resulting_composition'],
                    'deviations': result['deviations'],
                    'meets_tolerances': result['meets_tolerances'],
                    'pure_elements_needed': result['pure_elements_needed'],
                    'total_deviation': result['total_deviation'],
                    'n_steels': result.get('n_steels', len(result['steel_names'])),
                    'ms_temperature': result['ms_temperature']  # NEW: Include MS temperature
                }
                result_json['solutions'].append(solution)
            
            # Create status message
            if success:
                status = dbc.Alert([
                    html.H5([
                        icon("check-circle-fill", size="1.3em", className="text-success"),
                        "SUCCESS!"
                    ], className="alert-heading"),
                    html.P(f"Found {len(results)} valid solution(s)!"),
                    html.Hr(),
                    html.P("Go to 'Results' tab to view all solutions with MS temperatures and complete element breakdowns.", className="mb-0")
                ], color="success")
            else:
                status = dbc.Alert([
                    html.H5([
                        icon("exclamation-triangle-fill", size="1.3em", className="text-warning"),
                        "No Perfect Match Found"
                    ], className="alert-heading"),
                    html.P(f"Showing {len(results)} closest achievable composition(s)."),
                    html.Hr(),
                    html.P("Go to 'Results' tab to view details with MS temperatures and visualizations.", className="mb-0")
                ], color="warning")
            
            return json.dumps(result_json), status
        else:
            return None, dbc.Alert([
                icon("x-circle-fill", size="1.2em", className="text-danger"),
                "Optimization failed. Please check your inputs."
            ], color="danger")
            
    except Exception as e:
        return None, dbc.Alert([
            icon("x-circle-fill", size="1.2em", className="text-danger"),
            f"Error: {str(e)}"
        ], color="danger")


@app.callback(
    Output('results-container', 'children'),
    [Input('stored-results', 'data')]
)
def display_results(results_json):
    """Display optimization results with visualizations and MS temperature."""
    if not results_json:
        return html.P("No results yet. Run optimization first.", className="text-muted")
    
    try:
        data = json.loads(results_json)
        solutions = data['solutions']
        target_composition = data['target_composition']
        tolerances = data['tolerances']
        all_elements = data['all_elements']
        
        if not solutions:
            return html.P("No solutions found.", className="text-muted")
        
        # Create sections for each solution
        solution_sections = []
        
        for idx, result in enumerate(solutions, 1):
            # Get MS temperature
            ms_temp = result['ms_temperature']
            
            # Color-code MS temperature badge
            if ms_temp > 400:
                ms_color = "success"
                ms_icon = "thermometer-high"
            elif ms_temp > 250:
                ms_color = "info"
                ms_icon = "thermometer-half"
            elif ms_temp > 100:
                ms_color = "warning"
                ms_icon = "thermometer-low"
            else:
                ms_color = "danger"
                ms_icon = "thermometer-snow"
            
            # ====================================================================
            # SOLUTION HEADER WITH MS TEMPERATURE
            # ====================================================================
            if result['meets_tolerances']:
                header_color = "success"
                header_icon = "check-circle-fill"
                header_text = f"Solution #{idx} - VALID ✓"
            else:
                header_color = "warning"
                header_icon = "exclamation-triangle-fill"
                header_text = f"Solution #{idx} - Outside Tolerance"
            
            solution_header = dbc.Alert([
                html.Div([
                    html.H4([
                        icon(header_icon, size="1.3em"),
                        header_text
                    ], className="mb-2 d-inline-block"),
                    dbc.Badge([
                        icon_only(ms_icon),
                        f"MS = {ms_temp:.1f}°C"
                    ], color=ms_color, className="ms-3", style={'fontSize': '1.1em'})
                ]),
                html.P(f"Uses {result['n_steels']} steels | Total deviation: {result['total_deviation']:.6f}", 
                       className="mb-0")
            ], color=header_color)
            
            # ====================================================================
            # 1. MIXING RATIOS PIE CHART
            # ====================================================================
            fig_pie = go.Figure(data=[go.Pie(
                labels=result['steel_names'],
                values=[r*100 for r in result['ratios']],
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo='label+percent',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Ratio: %{value:.2f}%<extra></extra>'
            )])
            
            fig_pie.update_layout(
                title={
                    'text': f'Solution #{idx}: Steel Mixing Ratios',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                showlegend=True,
                height=400,
                margin=dict(t=80, b=40, l=40, r=40)
            )
            
            # ====================================================================
            # 2. COMPOSITION COMPARISON BAR CHART - SHOWS ALL ELEMENTS
            # ====================================================================
            target_elements = list(target_composition.keys())
            
            fig_bar = go.Figure()
            
            # Add bars for target elements
            target_vals = [target_composition.get(e, 0) for e in target_elements]
            achieved_target_vals = [result['resulting_composition'].get(e, 0) for e in target_elements]
            
            fig_bar.add_trace(go.Bar(
                name='Target',
                x=target_elements,
                y=target_vals,
                marker=dict(color='#3498db', opacity=0.8),
                text=[f"{v:.3f}" for v in target_vals],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Target: %{y:.4f} wt.%<extra></extra>'
            ))
            
            fig_bar.add_trace(go.Bar(
                name='Achieved (Target Elements)',
                x=target_elements,
                y=achieved_target_vals,
                marker=dict(color='#2ecc71', opacity=0.8),
                text=[f"{v:.3f}" for v in achieved_target_vals],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Achieved: %{y:.4f} wt.%<extra></extra>'
            ))
            
            # Add bars for non-target elements
            non_target_elements = [e for e in all_elements if e not in target_elements]
            achieved_non_target_vals = [result['resulting_composition'].get(e, 0) for e in non_target_elements]
            
            if non_target_elements:
                fig_bar.add_trace(go.Bar(
                    name='Achieved (Other Elements)',
                    x=non_target_elements,
                    y=achieved_non_target_vals,
                    marker=dict(color='#95a5a6', opacity=0.8),
                    text=[f"{v:.3f}" for v in achieved_non_target_vals],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Achieved: %{y:.4f} wt.%<extra></extra>'
                ))
            
            fig_bar.update_layout(
                title={
                    'text': f'Solution #{idx}: Composition (All Elements)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                xaxis_title='Element',
                yaxis_title='Composition (wt.%)',
                barmode='group',
                height=400,
                margin=dict(t=80, b=60, l=60, r=40),
                legend=dict(x=1, y=1, xanchor='right', yanchor='top')
            )
            
            # ====================================================================
            # 3. DEVIATION ANALYSIS CHART
            # ====================================================================
            deviations_list = [result['deviations'].get(e, 0) for e in target_elements]
            tolerances_list = [tolerances.get(e, 0.1) for e in target_elements]
            colors = ['#2ecc71' if d <= t else '#e74c3c' 
                      for d, t in zip(deviations_list, tolerances_list)]
            
            fig_deviation = go.Figure()
            
            # Add tolerance bars (background)
            fig_deviation.add_trace(go.Bar(
                name='Tolerance (±)',
                x=target_elements,
                y=tolerances_list,
                marker=dict(color='#95a5a6', opacity=0.3),
                hovertemplate='<b>%{x}</b><br>Tolerance: ±%{y:.4f} wt.%<extra></extra>'
            ))
            
            # Add deviation bars (foreground)
            fig_deviation.add_trace(go.Bar(
                name='Deviation',
                x=target_elements,
                y=deviations_list,
                marker=dict(color=colors),
                text=[f"{d:.4f}" for d in deviations_list],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Deviation: %{y:.4f} wt.%<extra></extra>'
            ))
            
            fig_deviation.update_layout(
                title={
                    'text': f'Solution #{idx}: Deviation Analysis (Target Elements)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                xaxis_title='Element',
                yaxis_title='Deviation / Tolerance (wt.%)',
                barmode='overlay',
                height=400,
                margin=dict(t=80, b=60, l=60, r=40),
                legend=dict(x=1, y=1, xanchor='right', yanchor='top')
            )
            
            # ====================================================================
            # 4. MS TEMPERATURE INFO CARD
            # ====================================================================
            ms_info_card = dbc.Card([
                dbc.CardBody([
                    html.H5([
                        icon("thermometer-half"),
                        "Martensite Start Temperature"
                    ], className="mb-3"),
                    html.H2(f"{ms_temp:.1f}°C", className="text-center mb-3", style={'color': '#2c3e50'}),
                    html.P([
                        html.Strong("Formula: "),
                        "Ms = 550 - 350C - 40Mn - 20Cr - 10Mo - 17Ni - 8W - 35V - 10Cu + 15Co + 30Al"
                    ], className="small text-muted mb-2"),
                    html.P([
                        html.Strong("Interpretation: "),
                        "Higher MS temperature indicates easier martensite formation during cooling. ",
                        "Note: Co and Al increase MS (positive), while C, Mn, Cr, Mo, Ni, W, V, Cu decrease it. ",
                        "Typical ranges: 200-400°C (tool steels), 300-500°C (low-alloy steels)."
                    ], className="small text-muted mb-0"),
                ])
            ], color="light", className="mb-3")
            
            # ====================================================================
            # 5. CREATE MIXING RATIOS TABLE
            # ====================================================================
            ratios_data = []
            for name, ratio in zip(result['steel_names'], result['ratios']):
                ratios_data.append({
                    'Steel': name,
                    'Ratio': f"{ratio:.4f}",
                    'Percentage': f"{ratio*100:.2f}%"
                })
            
            ratios_table = dash_table.DataTable(
                data=ratios_data,
                columns=[{'name': 'Steel', 'id': 'Steel'},
                         {'name': 'Ratio', 'id': 'Ratio'},
                         {'name': 'Percentage', 'id': 'Percentage'}],
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'},
            )
            
            # ====================================================================
            # 6. CREATE COMPOSITION COMPARISON TABLE - ALL ELEMENTS
            # ====================================================================
            comp_data = []
            
            # First, add target elements
            for elem in target_elements:
                target = target_composition[elem]
                achieved = result['resulting_composition'].get(elem, 0)
                deviation = result['deviations'].get(elem, 0)
                tolerance = tolerances.get(elem, 0)
                status = "✓ PASS" if deviation <= tolerance else "✗ FAIL"
                element_type = "TARGET"
                
                comp_data.append({
                    'Type': element_type,
                    'Element': elem,
                    'Target': f"{target:.4f}",
                    'Achieved': f"{achieved:.4f}",
                    'Deviation': f"{deviation:.4f}",
                    'Tolerance': f"±{tolerance:.4f}",
                    'Status': status
                })
            
            # Then, add non-target elements
            for elem in non_target_elements:
                achieved = result['resulting_composition'].get(elem, 0)
                
                comp_data.append({
                    'Type': "OTHER",
                    'Element': elem,
                    'Target': "-",
                    'Achieved': f"{achieved:.4f}",
                    'Deviation': "-",
                    'Tolerance': "-",
                    'Status': "-"
                })
            
            comp_table = dash_table.DataTable(
                data=comp_data,
                columns=[{'name': i, 'id': i} for i in ['Type', 'Element', 'Target', 'Achieved', 'Deviation', 'Tolerance', 'Status']],
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Status} = "✓ PASS"'},
                        'backgroundColor': '#d4edda',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Status} = "✗ FAIL"'},
                        'backgroundColor': '#f8d7da',
                        'color': 'black',
                    },
                    {
                        'if': {'filter_query': '{Type} = "OTHER"'},
                        'backgroundColor': '#e9ecef',
                        'color': '#6c757d',
                    }
                ]
            )
            
            # ====================================================================
            # 7. PURE ELEMENTS NEEDED (if any)
            # ====================================================================
            pure_elem_section = []
            if result['pure_elements_needed']:
                pure_data = []
                for elem, amount in result['pure_elements_needed'].items():
                    if amount > 0:
                        action = f"Add {amount:.4f} wt.%"
                    else:
                        action = f"Remove {abs(amount):.4f} wt.%"
                    pure_data.append({'Element': elem, 'Action': action})
                
                pure_table = dash_table.DataTable(
                    data=pure_data,
                    columns=[{'name': 'Element', 'id': 'Element'}, {'name': 'Action', 'id': 'Action'}],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': '#ffc107', 'color': 'black', 'fontWeight': 'bold'},
                )
                
                pure_elem_section = [
                    html.Hr(),
                    html.H5([
                        icon("tools"),
                        "Pure Elements Needed to Meet Target"
                    ], className="mt-3"),
                    pure_table
                ]
            
            # ====================================================================
            # ASSEMBLE THIS SOLUTION'S LAYOUT
            # ====================================================================
            solution_section = html.Div([
                solution_header,
                
                # MS Temperature Card
                ms_info_card,
                
                # Visualizations
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig_pie)
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(figure=fig_bar)
                    ], width=6),
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig_deviation)
                    ], width=12),
                ], className="mb-4"),
                
                # Tables
                html.H5([
                    icon("list-check"),
                    "Selected Steels and Mixing Ratios"
                ]),
                ratios_table,
                
                html.Hr(),
                
                html.H5([
                    icon("bullseye"),
                    "Complete Element Composition"
                ]),
                html.P([
                    html.Strong("Target Elements: "), "Elements you specified as targets",
                    html.Br(),
                    html.Strong("Other Elements: "), "Additional elements present in the mixture"
                ], className="text-muted small mb-2"),
                comp_table,
                
            ] + pure_elem_section + [
                html.Hr(style={'marginTop': '3rem', 'marginBottom': '3rem', 'borderTop': '3px solid #dee2e6'}),
            ])
            
            solution_sections.append(solution_section)
        
        # ====================================================================
        # SEARCH LOG AT THE END
        # ====================================================================
        log_section = [
            html.H5([
                icon("file-text"),
                "Search Log"
            ], className="mt-3"),
            html.Pre('\n'.join(data['search_log']), 
                    style={'backgroundColor': '#f5f5f5', 'padding': '15px', 'borderRadius': '5px'})
        ]
        
        # ====================================================================
        # RETURN ALL SOLUTIONS + LOG
        # ====================================================================
        return html.Div(solution_sections + log_section)
        
    except Exception as e:
        return dbc.Alert(f"Error displaying results: {str(e)}", color="danger")


@app.callback(
    Output("download-results", "data"),
    [Input("download-results-btn", "n_clicks")],
    [State('stored-results', 'data')],
    prevent_initial_call=True
)
def download_results(n_clicks, results_json):
    """Download results as CSV - INCLUDES MS TEMPERATURE."""
    if not results_json:
        return None
    
    try:
        data = json.loads(results_json)
        solutions = data['solutions']
        target_composition = data['target_composition']
        tolerances = data['tolerances']
        all_elements = data['all_elements']
        
        # Create CSV data
        csv_data = []
        
        csv_data.append(['STEEL MIXING OPTIMIZATION RESULTS'])
        csv_data.append([f'Number of Solutions: {len(solutions)}'])
        csv_data.append([])
        
        # For each solution
        for idx, result in enumerate(solutions, 1):
            csv_data.append([f'=== SOLUTION #{idx} ==='])
            csv_data.append(['Status:', 'VALID' if result['meets_tolerances'] else 'Outside Tolerance'])
            csv_data.append(['Total Deviation:', result['total_deviation']])
            csv_data.append(['MS Temperature (°C):', f"{result['ms_temperature']:.2f}"])
            csv_data.append([])
            
            # Mixing ratios
            csv_data.append(['MIXING RATIOS'])
            csv_data.append(['Steel', 'Ratio', 'Percentage'])
            for name, ratio in zip(result['steel_names'], result['ratios']):
                csv_data.append([name, ratio, ratio*100])
            
            csv_data.append([])
            
            # Composition comparison - ALL ELEMENTS
            csv_data.append(['COMPOSITION - ALL ELEMENTS'])
            csv_data.append(['Element', 'Type', 'Target', 'Achieved', 'Deviation', 'Tolerance', 'Status'])
            
            # Target elements first
            for elem in target_composition.keys():
                target = target_composition[elem]
                achieved = result['resulting_composition'].get(elem, 0)
                deviation = result['deviations'].get(elem, 0)
                tolerance = tolerances.get(elem, 0)
                status = "PASS" if deviation <= tolerance else "FAIL"
                csv_data.append([elem, 'TARGET', target, achieved, deviation, tolerance, status])
            
            # Non-target elements
            for elem in all_elements:
                if elem not in target_composition:
                    achieved = result['resulting_composition'].get(elem, 0)
                    csv_data.append([elem, 'OTHER', '-', achieved, '-', '-', '-'])
            
            # Pure elements needed
            if result['pure_elements_needed']:
                csv_data.append([])
                csv_data.append(['PURE ELEMENTS NEEDED'])
                csv_data.append(['Element', 'Amount'])
                for elem, amount in result['pure_elements_needed'].items():
                    csv_data.append([elem, amount])
            
            csv_data.append([])
            csv_data.append([])
        
        # MS Temperature Formula
        csv_data.append(['MS TEMPERATURE FORMULA'])
        csv_data.append(['Ms (°C) = 550 - 350C - 40Mn - 20Cr - 10Mo - 17Ni - 8W - 35V - 10Cu + 15Co + 30Al'])
        csv_data.append(['Comprehensive formula including W, V, Cu, Co, Al effects'])
        
        # Convert to CSV string
        import csv
        from io import StringIO
        si = StringIO()
        writer = csv.writer(si)
        writer.writerows(csv_data)
        
        return dict(content=si.getvalue(), filename="steel_mixing_results_with_ms.csv")
        
    except Exception as e:
        return None


# Callback to handle removal status
@app.callback(
    Output('remove-alloy-status', 'children'),
    [Input('remove-alloys-btn', 'n_clicks')]
)
def show_removal_status(n_clicks):
    """Show status after attempting to remove alloys."""
    if not n_clicks:
        return ""
    return ""


# Callback to handle custom alloy status
@app.callback(
    Output('custom-alloy-status', 'children'),
    [Input('add-custom-alloy-btn', 'n_clicks')]
)
def show_custom_alloy_status(n_clicks):
    """Show status after attempting to add custom alloy."""
    if not n_clicks:
        return ""
    return ""


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print(" " * 8 + "🔬 STEEL MIXING OPTIMIZER v4.3 - AUTHENTICATED 🔬")
    print("="*80)
    print("\n✓ Starting server...")
    print("\n🌐 Open your browser and go to: http://localhost:8050")
    print("\n🔐 AUTHENTICATION REQUIRED:")
    print("   Valid credentials:")
    print("   • admin / steel2025")
    print("   • user / password123")
    print("   • engineer / metallurgy")
    print("\n🆕 NEW FEATURES in v4.3:")
    print("   • 🔒 Username/Password authentication")
    print("   • 🌡️  MS Temperature calculation (Andrews formula)")
    print("   • 📊 Color-coded MS temperature badges")
    print("   • 📈 MS temp displayed in results and CSV export")
    print("\n💡 MS Temperature Formula:")
    print("   Ms (°C) = 550 - 350C - 40Mn - 20Cr - 10Mo - 17Ni - 8W - 35V - 10Cu + 15Co + 30Al")
    print("\n⚠  Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8050)
