from flask import request, render_template, redirect, url_for, session,flash, jsonify
import pandas as pd
import plotly.express as px
import os
from init import app
from app import generate_narrative
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to check and convert columns to numeric where possible
def convert_to_numeric(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                df[column] = pd.to_numeric(df[column])
            except ValueError:
                continue
    return df

def check_data_exists():
    return os.path.exists('uploads/cleaned_initial_responses.xlsx')

@app.route('/')
def index():
    no_data = not check_data_exists()
    return render_template('index.html', no_data=no_data)


@app.route('/analyze_page/<report_type>')
def analyze_page(report_type):
    filepath = session.get('filepath')
    if not filepath:
        return jsonify(success=False, message='No file uploaded yet.')

    filename = os.path.basename(filepath)
    if report_type == 'employment_status':
        return jsonify(success=True, url=url_for('analyze', filename=filename))
    elif report_type == 'trend_analysis':
        return jsonify(success=True, url=url_for('analyze_trend', filename=filename))
    elif report_type == 'predictive_insights':
        return jsonify(success=True, url=url_for('analyze_predictive', filename=filename))
    else:
        return jsonify(success=False, message='Invalid report type.')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(success=False)
        file = request.files['file']
        if file.filename == '':
            return jsonify(success=False)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['filepath'] = filepath
            data = pd.read_excel(filepath)
            
            # Data cleaning
            data.fillna(method='ffill', inplace=True)
            data.drop_duplicates(inplace=True)
            data['Date Started'] = pd.to_datetime(data['Date Started'], errors='coerce').dt.date
            data['Date Graduated'] = pd.to_datetime(data['Date Graduated'], errors='coerce').dt.date
            data['Employment Status'] = data['Employment Status'].str.title()
            
            cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_' + file.filename)
            data.to_excel(cleaned_filepath, index=False)
            session['filepath'] = cleaned_filepath
            return jsonify(success=True, filename='cleaned_' + file.filename)
    return render_template('upload.html')

# Employment status
@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = session.get('filepath')
    if not filepath:
        flash('No file uploaded yet. Please upload a file to analyze.', 'warning')
        return redirect(url_for('upload_file'))

    try:
        data = pd.read_excel(filepath)
    except FileNotFoundError:
        flash('The specified file does not exist. Please upload a valid file.', 'error')
        return redirect(url_for('upload_file'))

    # Extract and format date range
    min_date_started = data['Date Started'].min().strftime('%B %d, %Y')
    max_date_graduated = data['Date Graduated'].max().strftime('%B %d, %Y')

    employment_status_dist = data.groupby(['Course Completed', 'Employment Status']).size().unstack().fillna(0)
    job_title_dist = data.groupby(['Course Completed', 'Job Title']).size().unstack().fillna(0)
    industry_type_dist = data.groupby(['Course Completed', 'Industry Type']).size().unstack().fillna(0)
    income_dist = data.groupby(['Course Completed', 'Monthly Income Range']).size().unstack().fillna(0)
    job_satisfaction_avg = data.groupby('Course Completed')['Job Satisfaction'].mean()

    employment_status_fig = px.bar(employment_status_dist, title='Employment Status Distribution by Course')
    job_title_fig = px.bar(job_title_dist, title='Job Title Distribution by Course')
    industry_type_fig = px.bar(industry_type_dist, title='Industry Type Distribution by Course')
    income_dist_fig = px.bar(income_dist, title='Income Distribution by Course')
    job_satisfaction_fig = px.bar(job_satisfaction_avg, title='Average Job Satisfaction by Course')

    employment_status_html = employment_status_fig.to_html(full_html=False)
    job_title_html = job_title_fig.to_html(full_html=False)
    industry_type_html = industry_type_fig.to_html(full_html=False)
    income_dist_html = income_dist_fig.to_html(full_html=False)
    job_satisfaction_html = job_satisfaction_fig.to_html(full_html=False)

    narratives = generate_narrative(data)

    return render_template('analysis.html',
                           employment_status_html=employment_status_html,
                           job_title_html=job_title_html,
                           industry_type_html=industry_type_html,
                           income_dist_html=income_dist_html,
                           job_satisfaction_html=job_satisfaction_html,
                           narratives=narratives,
                           min_date_started=min_date_started,
                           max_date_graduated=max_date_graduated)


# Trend Analysis Route with Stacked Area Chart
@app.route('/analyze_trend/<filename>')
def analyze_trend(filename):
    filepath = session.get('filepath')
    if not filepath:
        flash('No file uploaded yet. Please upload a file to analyze.', 'warning')
        return redirect(url_for('upload_file'))

    try:
        data = pd.read_excel(filepath)
    except FileNotFoundError:
        flash('The specified file does not exist. Please upload a valid file.', 'error')
        return redirect(url_for('upload_file'))

    if 'Date Graduated' in data.columns:
        data['Date Graduated'] = pd.to_datetime(data['Date Graduated'], errors='coerce')
        data['Graduation Year'] = data['Date Graduated'].dt.year
    else:
        flash('Date Graduated column not found in the uploaded file.', 'error')
        return redirect(url_for('upload_file'))

    data = data.dropna(subset=['Graduation Year'])

    grad_year_trend = data.groupby(['Graduation Year', 'Course Completed']).size().unstack().fillna(0)

    trend_fig = px.area(grad_year_trend, title='Employment Trends Over Time by Graduation Year and Course')
    trend_html = trend_fig.to_html(full_html=False)

    narratives = generate_narrative_trend(data)

    return render_template('trend_analysis.html', trend_html=trend_html, narratives=narratives)

def generate_narrative_trend(data):
    narratives = {}

    grad_years = sorted(data['Graduation Year'].unique())
    courses = data['Course Completed'].unique()

    narrative = "The trend analysis shows the following insights:<br>"
    course_summaries = []

    course_yearly_summary = {course: {year: {'graduates': 0, 'employed': 0, 'self_employed': 0, 'unemployed': 0} for year in grad_years} for course in courses}

    for course in courses:
        course_narrative = f"<br><strong>For the course '{course}':</strong><br>"
        for year in grad_years:
            course_data = data[(data['Graduation Year'] == year) & (data['Course Completed'] == course)]
            if not course_data.empty:
                employed = len(course_data[course_data['Employment Status'] == 'Employed'])
                self_employed = len(course_data[course_data['Employment Status'] == 'Self-Employed'])
                unemployed = len(course_data[course_data['Employment Status'] == 'Unemployed'])
                total_graduates = len(course_data)
                course_narrative += f"  - In {year}: {total_graduates} graduates ({employed} employed, {self_employed} self-employed, {unemployed} unemployed).<br>"
                course_yearly_summary[course][year]['graduates'] = total_graduates
                course_yearly_summary[course][year]['employed'] = employed
                course_yearly_summary[course][year]['self_employed'] = self_employed
                course_yearly_summary[course][year]['unemployed'] = unemployed
        narrative += course_narrative

    overall_summary = "<br><strong>Overall Summary of Trends:</strong><br>"
    yearly_top_courses = []

    for year in grad_years:
        top_course = max(course_yearly_summary.items(), key=lambda x: x[1][year]['graduates'])[0]
        top_course_employment = course_yearly_summary[top_course][year]
        yearly_top_courses.append(f"In {year}, the top course was '{top_course}' with {top_course_employment['graduates']} graduates ({top_course_employment['employed']} employed, {top_course_employment['self_employed']} self-employed, {top_course_employment['unemployed']} unemployed).<br>")

    overall_summary += "".join(yearly_top_courses)

    narratives['trend_analysis'] = narrative + overall_summary
    return narratives

# Predictive Analysis Route
import matplotlib.pyplot as plt
import seaborn as sns

# Predictive Analysis Route
@app.route('/analyze_predictive/<filename>')
def analyze_predictive(filename):
    filepath = session.get('filepath')
    print(f"Session Filepath Retrieved: {filepath}")

    if not filepath:
        flash('No file uploaded yet. Please upload a file to analyze.', 'warning')
        return redirect(url_for('upload_file'))

    if not os.path.exists(filepath):
        flash('The specified file does not exist. Please upload a valid file.', 'error')
        return redirect(url_for('upload_file'))

    try:
        data = pd.read_excel(filepath)
        print("Original DataFrame:")
        print(data.head())
    except Exception as e:
        flash(f"Error reading the Excel file: {e}", 'error')
        return redirect(url_for('upload_file'))

    try:
        data.fillna({
            'Gender': data['Gender'].mode()[0],
            'Age': data['Age'].mean(),
            'Civil Status': data['Civil Status'].mode()[0],
            'Prior Education': data['Prior Education'].mode()[0],
            'Technical Skills': data['Technical Skills'].mode()[0],
            'Soft Skills': data['Soft Skills'].mode()[0],
            'Leadership Roles': data['Leadership Roles'].mode()[0],
            'Type of Organization': data['Type of Organization'].mode()[0],
            'Industry Type': data['Industry Type'].mode()[0],
            'Job Title': data['Job Title'].mode()[0],
            'Employment Status': data['Employment Status'].mode()[0],
            'Job Satisfaction': data['Job Satisfaction'].mean() if 'Job Satisfaction' in data else 0
        }, inplace=True)

        categorical_columns = ['Gender', 'Civil Status', 'Prior Education', 'Course Completed', 'Specialization', 
                               'Type of Organization', 'Industry Type', 'Job Title', 'Employment Status']
        data_encoded = pd.get_dummies(data, columns=categorical_columns, dummy_na=True)

        boolean_columns = data_encoded.columns[data_encoded.columns.str.contains('Employment Status_')]
        data_encoded[boolean_columns] = data_encoded[boolean_columns].astype(int)

        numerical_columns = ['Age', 'Job Satisfaction']
        data_encoded[numerical_columns] = data_encoded[numerical_columns].apply(lambda x: (x - x.mean()) / x.std())

        data_encoded = data_encoded.select_dtypes(include=[np.number])

        print("Encoded DataFrame Columns:", data_encoded.columns.tolist())
        print("Encoded DataFrame Head:", data_encoded.head())

        if 'Employment Status_Employed' in data_encoded:
            target_employment = data_encoded['Employment Status_Employed']
        else:
            print("Employment Status_Employed column is missing.")
            target_employment = None

        target_satisfaction = data_encoded['Job Satisfaction']

        features = data_encoded.drop(columns=['Employment Status_Employed', 'Job Satisfaction'], errors='ignore')

        if features.empty:
            flash('No features available for model training.', 'error')
            return redirect(url_for('upload_file'))

        X_train_emp, X_test_emp, y_train_emp, y_test_emp = train_test_split(features, target_employment, test_size=0.2, random_state=42)
        X_train_sat, X_test_sat, y_train_sat, y_test_sat = train_test_split(features, target_satisfaction, test_size=0.2, random_state=42)

        model_emp = LogisticRegression(max_iter=1000)
        model_emp.fit(X_train_emp, y_train_emp)
        y_pred_emp = model_emp.predict(X_test_emp)
        accuracy_emp = accuracy_score(y_test_emp, y_pred_emp)
        print(f"Employment Model Accuracy: {accuracy_emp}")

        model_sat = LinearRegression()
        model_sat.fit(X_train_sat, y_train_sat)
        y_pred_sat = model_sat.predict(X_test_sat)
        mse_sat = mean_squared_error(y_test_sat, y_pred_sat)
        print(f"Job Satisfaction Model MSE: {mse_sat}")

        insights, recommendations, employment_trends_fig = generate_insights_and_recommendations(data, model_emp, model_sat, accuracy_emp, mse_sat)
        print("Insights and Recommendations Generated:", insights, recommendations)

        # Save the job satisfaction by course plot
        satisfaction_by_course = data.groupby('Course Completed')['Job Satisfaction'].mean().round(2)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=satisfaction_by_course.index, y=satisfaction_by_course.values, palette='viridis')
        plt.title('Average Job Satisfaction by Course')
        plt.xlabel('Course Completed')
        plt.ylabel('Average Job Satisfaction')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/average_job_satisfaction_by_course.png')

    except Exception as e:
        flash(f"Error processing the data: {e}", 'error')
        print(f"Exception during processing: {e}")
        return redirect(url_for('upload_file'))

    try:
        return render_template('predictive_analysis.html', insights=insights, recommendations=recommendations, employment_trends_fig=employment_trends_fig)
    except Exception as e:
        flash(f"Error rendering template: {e}", 'error')
        print(f"Exception during template rendering: {e}")
        return redirect(url_for('upload_file'))



def generate_insights_and_recommendations(data, model_emp, model_sat, accuracy_emp, mse_sat):
    insights = {}
    recommendations = {}

    # Insights for Employment Status
    insights['employment'] = f"Logistic Regression Model Accuracy for Employment Status: {accuracy_emp:.2f}"
    
    # Consider both 'Employed' and 'Self-Employed' as employed
    employment_by_course = data.groupby('Course Completed')['Employment Status'].apply(
        lambda x: ((x == 'Employed') | (x == 'Self-Employed')).mean()
    )
    insights['employment_by_course'] = employment_by_course.to_dict()

    # Identify courses with lower employment rates
    low_employment_courses = employment_by_course[employment_by_course < 0.8].index.tolist()
    recommendations['employment'] = "Improve curriculum for courses with lower employment rates. Consider adding practical training, industry partnerships, or soft skills development."
    if low_employment_courses:
        recommendations['employment_details'] = f"The following courses have lower employment rates: {', '.join(low_employment_courses)}."
        recommendations['employment_reasons'] = {course: f"Employment Rate: {employment_by_course[course] * 100:.2f}%" for course in low_employment_courses}

    # Detailed breakdown for each course
    course_details = {
        course: f"{(data['Course Completed'] == course).sum()} total students, {((data['Course Completed'] == course) & ((data['Employment Status'] == 'Employed') | (data['Employment Status'] == 'Self-Employed'))).sum()} employed (including self-employed), Employment Rate: {employment_by_course[course] * 100:.2f}%"
        for course in data['Course Completed'].unique()
    }
    recommendations['employment_course_details'] = course_details

    # Insights for Job Satisfaction
    insights['satisfaction'] = f"Linear Regression Model MSE for Job Satisfaction: {mse_sat:.2f}"
    satisfaction_by_course = data.groupby('Course Completed')['Job Satisfaction'].mean()
    insights['satisfaction_by_course'] = satisfaction_by_course.to_dict()

    # Identify courses with lower job satisfaction
    low_satisfaction_courses = satisfaction_by_course[satisfaction_by_course < satisfaction_by_course.mean()].index.tolist()
    recommendations['satisfaction'] = "Enhance course content for courses with lower job satisfaction. Include more hands-on projects and real-world applications."
    if low_satisfaction_courses:
        recommendations['satisfaction_details'] = f"The following courses have lower job satisfaction: {', '.join(low_satisfaction_courses)}."
        recommendations['satisfaction_reasons'] = {course: f"Average Job Satisfaction: {satisfaction_by_course[course]:.2f}" for course in low_satisfaction_courses}

    # Generate employment trends visualization
    employment_trends_fig = px.line(employment_by_course.reset_index(), x='Course Completed', y='Employment Status',
                                    title='Employment Trends by Course', labels={'Employment Status': 'Employment Rate'})
    employment_trends_html = employment_trends_fig.to_html(full_html=False)

    return insights, recommendations, employment_trends_html



if __name__ == '__main__':
    app.run(debug=True)








