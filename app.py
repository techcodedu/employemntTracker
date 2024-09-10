import pandas as pd

def generate_narrative(data):
    narratives = {}

    # Extract date range from the data
    date_started = pd.to_datetime(data['Date Started'])
    date_graduated = pd.to_datetime(data['Date Graduated'])
    date_range = f"{date_started.min().strftime('%B %Y')} to {date_graduated.max().strftime('%B %Y')}"

    narratives['date_range'] = f"The data covers the period from {date_range}."

    # Employment Status Narrative
    employment_status_count = data['Employment Status'].value_counts()
    total_graduates = len(data)
    employed_percentage = (employment_status_count.get('Employed', 0) / total_graduates) * 100
    self_employed_percentage = (employment_status_count.get('Self-Employed', 0) / total_graduates) * 100
    unemployed_percentage = (employment_status_count.get('Unemployed', 0) / total_graduates) * 100
    narratives['employment_status'] = (
        f"Out of {total_graduates} graduates, {employment_status_count.get('Employed', 0)} ({employed_percentage:.2f}%) are employed, "
        f"{employment_status_count.get('Self-Employed', 0)} ({self_employed_percentage:.2f}%) are self-employed, "
        f"and {employment_status_count.get('Unemployed', 0)} ({unemployed_percentage:.2f}%) are unemployed."
    )

    # Analyze the need for curriculum improvement
    unemployed_by_course = data[data['Employment Status'] == 'Unemployed']['Course Completed'].value_counts()
    total_by_course = data['Course Completed'].value_counts()
    for course, total in total_by_course.items():
        unemployed = unemployed_by_course.get(course, 0)
        unemployment_rate = (unemployed / total) * 100
        if unemployment_rate > 20:  # Example threshold for high unemployment rate
            narratives['employment_status'] += (
                f"\n\nThe course '{course}' has a high unemployment rate of {unemployment_rate:.2f}%, indicating a potential need for curriculum improvement."
            )

    # Job Title Narrative
    job_title_count = data['Job Title'].value_counts()
    narratives['job_title'] = (
        f"The most common job title among graduates is '{job_title_count.idxmax()}' with {job_title_count.max()} graduates holding this position. "
        f"This indicates a strong preference or demand for this role among employers."
    )

    # Industry Type Narrative
    industry_type_count = data['Industry Type'].value_counts()
    narratives['industry_type'] = (
        f"The predominant industry type among graduates is '{industry_type_count.idxmax()}', which employs {industry_type_count.max()} graduates. "
        f"This suggests that the {industry_type_count.idxmax()} industry is a significant sector for employment opportunities."
    )

    # Income Range Narrative
    income_range_count = data['Monthly Income Range'].value_counts()
    narratives['income'] = (
        f"The most common income range is '{income_range_count.idxmax()}', with {income_range_count.max()} graduates falling into this category. "
        f"This highlights the earning potential for most graduates."
    )

    # Job Satisfaction Narrative
    job_satisfaction_avg = data['Job Satisfaction'].mean()
    highest_satisfaction_course = data.groupby('Course Completed')['Job Satisfaction'].mean().idxmax()
    highest_satisfaction_score = data.groupby('Course Completed')['Job Satisfaction'].mean().max()
    narratives['job_satisfaction'] = (
        f"The average job satisfaction score among graduates is {job_satisfaction_avg:.2f} out of 5. "
        f"The course with the highest job satisfaction is '{highest_satisfaction_course}' with an average score of {highest_satisfaction_score:.2f}, "
        f"indicating that graduates from this course felt they learned a lot and are highly satisfied with their job positions."
    )

    return narratives
