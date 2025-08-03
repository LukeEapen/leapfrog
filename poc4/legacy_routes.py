# This file is legacy and not used by the current backend. See poc4/backend/routes.py for active routes.
from flask import Blueprint, render_template, request, redirect, url_for, session
from agents import (
    schema_mapping_agent,
    transformation_rule_agent,
    validation_agent,
    migration_execution_agent,
    reconciliation_agent,
    chatbot_agent
)

poc4_bp = Blueprint('poc4', __name__, url_prefix='/poc4')

@poc4_bp.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        # Save uploaded files and selections to session or DB
        # Call schema_mapping_agent to analyze schemas
        return redirect(url_for('poc4.page2'))
    return render_template('poc4/page1_upload.html')

@poc4_bp.route('/page2', methods=['GET', 'POST'])
def page2():
    if request.method == 'POST':
        # Save mapping edits
        # Call transformation_rule_agent for suggestions
        return redirect(url_for('poc4.page3'))
    # Get auto-mapping from schema_mapping_agent
    return render_template('poc4/page2_mapping.html')

@poc4_bp.route('/page3', methods=['GET', 'POST'])
def page3():
    if request.method == 'POST':
        # Save transformation rules
        # Call validation_agent to simulate migration
        return redirect(url_for('poc4.page4'))
    # Get suggested rules from transformation_rule_agent
    return render_template('poc4/page3_rules.html')

@poc4_bp.route('/page4', methods=['GET', 'POST'])
def page4():
    if request.method == 'POST':
        # Run migration_execution_agent
        return redirect(url_for('poc4.page5'))
    # Show validation results from validation_agent
    return render_template('poc4/page4_validation.html')

@poc4_bp.route('/page5', methods=['GET', 'POST'])
def page5():
    if request.method == 'POST':
        # Approve/export results
        pass
    # Show reconciliation report from reconciliation_agent
    return render_template('poc4/page5_reconciliation.html')

@poc4_bp.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    response = chatbot_agent.handle_message(user_message, session)
    return {'response': response}
