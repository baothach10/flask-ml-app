# Import the resources and libraries
import os
import numpy as np
import pickle
from flask import Flask, redirect, render_template, request
from nltk.probability import *
from gensim.models.fasttext import FastText


# This is a function to convert a txt job file to a dictionary with keys are the items
def convert_txt_to_dict(file_path, folder_name):
    # Get the ID of the job advertisement
    job_id = (file_path.split('Job_')[-1]).split('.')[0]
    # Open the file for reading
    file = open(file_path, 'r')
    # Create an empty dictionary to store all items of a job txt file
    job_dict = {}
    # Assign the job id
    job_dict['ID'] = job_id
    # Assign the folder name of the job advetisement
    job_dict['Category'] = folder_name
    # Loop through each line in the txt file
    for line in file:
        # Split the line by the colon
        line_data = line.split(':')
        # Take the first element in the splitted list since it is the item and remove extra spaces
        item = line_data[0].strip()
        # Check if it is the description of the file then we have a different way to deal with it since it is more complex
        if item == 'Description':
            # Concatenate all elements in the splitted string by the colon since there can be colons in the text
            job_dict[item] = ':'.join(line_data[1:]).strip()
        else:
            # Remove the newline tag from the item before assigning the key and value to the dictionary
            cleaned_string = ''.join(line_data[1:]).split('\n')
            job_dict[item] = ''.join(cleaned_string).strip()
    # Close the file
    file.close()
    # Return the dictionary 
    return job_dict


# This is a function to extract the data via the root path of the data folder and add new job information into a list
def load_data(root_path, data, folder=''):
    # Get a list of files via the given path and remain the same order of the files
    file_list = sorted(os.listdir(root_path))
    # Iterate through each file and print out the file name and size if it is a file, if it is a folder, recursively execute this function
    for file in file_list:
        # Ignore the first component of the folder since it is irrelevant
        if file == '.DS_Store':
            continue
        # Combine the path of the file
        file_path = os.path.join(root_path, file)
        # Check if it is a file
        if os.path.isfile(file_path):
            # Append the job into the data list after converting it into a dictionary
            data.append(convert_txt_to_dict(file_path, folder))
        # Check if the file is a folder
        elif os.path.isdir(file_path):
            # Use the function recursively to print out all the files of all folders
            load_data(file_path, data, file)
    return data

# This is a function to save the data into the file directory
def save_data(job_dict, root_path='./data'):
    # Create the folder path of the current job dictionary
    folder_path =  os.path.join(root_path, job_dict['Category'])
    # If it does not exist the path then create a new folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Create a new file name
    text_file_name = 'Job_' + job_dict['ID'] + '.txt'
    # Create the path for the file
    text_file_path = os.path.join(folder_path, text_file_name)
    # Open the file for writing
    file = open(text_file_path, 'w')
    # Write into the file
    for field in job_dict.keys():
        # Craft the data string
        data_string =  field + ': ' + job_dict[field] + '\n'
        # Write into the file
        file.write(data_string)
    # Close the file
    file.close()


# Generate unweighted vector representations
def docvecs(embeddings, docs):
    # Initialize an array to store document vectors
    vecs = np.zeros((len(docs), embeddings.vector_size))
    # Loop through each document in the list of job descriptions
    for i, doc in enumerate(docs):
        # Filter out terms in the document that are present in the embeddings
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        # Check if valid keys are empty
        if valid_keys:
            # Initialize an array to store the vectors of valid terms in the document
            docvec = np.vstack([embeddings[term] for term in valid_keys])
            # Calculate the document vector by summing the vectors of valid terms
            docvec = np.sum(docvec, axis=0)
            # Store the document vector in the 'vecs' array
            vecs[i,:] = docvec
    return vecs
        

# Define the FLask application
app = Flask(__name__)
            
# Define the data folder directory
data_folder_path = './data'
# Create an empty list
data = []
# Load the data using the function
load_data(data_folder_path, data)

# Route for the home page, displaying the jobs for job seeker
# This is also the job seekers site
@app.route('/index.html')
@app.route('/')
def index():
    # Render the template for the job display page
    return render_template('index.html', jobs = data)

# Route for the employer page, creating new job listing
@app.route('/employer.html')
def enter_job():
    # Render the template for the employer page
    return render_template('employer.html')

# Route for the job detail, displaying all detailed information of each job
@app.route('/jobDetail/<int:job_id>')
def display_job_detail(job_id):
    # Iterate through each job in the list to find the correct one to display
    job_dict = {}
    for job in data:
        if int(job['ID']) == int(job_id):
            job_dict = job
            break
    # Render the template with the specific job details
    return render_template('jobDetail.html', job_dict=job_dict)

# Route for filtering the jobs based on category
@app.route('/filter_job', methods=['GET'])
def filter_job():
    # Get the selected category via the repsonse of the form from the front end
    selected_category = request.args.get('selected_category')
    selected_jobs = []
    # If the option is 'All' then display all of them
    if selected_category == 'All':
        return render_template('index.html', jobs = data)
    else:
        # Iterate through each job to display only the ones with the selected category
        for job in data:
            if job['Category'] == selected_category:
                selected_jobs.append(job)
        # Render the job display page with only the filtered jobs
        return render_template('index.html', jobs = selected_jobs)
    
# Route for searching for a specific job based on a keyword
@app.route('/search', methods=['GET'])
def search():
    # Get the keyword from the form repsonse of the front end based on any fields of a job
    keywords = request.args.get('keyword')
    matched_jobs = []
    # Iterate through each job
    for job in data:
        # Iterate through each field of a job to check if there are any matches
        for field in job.keys():
            # If there is a match, add that job into a list and move to the next job to check
            if keywords.lower() in str(job[field]).lower():
                matched_jobs.append(job)
                break
    # Render the job display page with only the jobs with matched information
    return render_template('index.html', jobs = matched_jobs)

# Route for creating new job listing
@app.route('/create_job', methods=['POST'])
def create():
    job_dict = {}
    # Get the title from the form response of the front end and add it to the dictionary
    title = request.form['job_title']
    job_dict['Title'] = title
    # Get the name of the company from the form response of the front end and add it to the dictionary
    company = request.form['company']
    job_dict['Company'] = company
    # Get the description from the form response of the front end and add it to the dictionary
    description = request.form['job_description']
    job_dict['Description'] = description
    # Get the salary from the form response of the front end and add it to the dictionary
    salary = request.form['salary']
    job_dict['Salary'] = salary
    # Get the other information from the form response of the front end and add it to the dictionary
    other_info_label = request.form['other_information_label'].capitalize()
    other_info = request.form['other_information']
    job_dict[other_info_label] = other_info
    # Tokenize the data
    tokenized_data = description.split(' ')
    # Load the pre-trained FastText model from milestone I
    job_ads_FT = FastText.load('job_ads_FT.model')
    # Create word vectors
    job_ads_wv = job_ads_FT.wv
    # Create unweighted vector representation for the tokenized data
    job_ads_dvs = docvecs(job_ads_wv, [tokenized_data])
    # Load the pre-trained Logistic Regression model from the milestone I
    with open('job_ads_FT_LR_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    # Predict the category from the unweighted vector representation
    y_pred = model.predict(job_ads_dvs)
    predicted_category = y_pred[0]
    # Render the template to display the input information and the predicted category for the user to confirm
    return render_template('jobDetail_final_page.html', job_dict = job_dict, other_info_label = other_info_label, predicted_category = predicted_category)

# Route for save the job
@app.route('/save_job', methods=['POST'])
def save():
    job_dict = {}
    # Create the job ID based on the current number of files to make every ID unique
    job_dict['ID'] = "{:05d}".format(len(data) + 1)
    # Get the title from the form response of the front end and add it to the dictionary
    title = request.form['job_title']
    job_dict['Title'] = title
    # Get the predicted category from the form response of the front end and add it to the dictionary
    category = request.form['category']
    job_dict['Category'] = category
    # Get the name of the company from the form response of the front end and add it to the dictionary
    company = request.form['company']
    job_dict['Company'] = company
    # Get the description from the form response of the front end and add it to the dictionary
    description = request.form['job_description']
    job_dict['Description'] = description
    # Get the salary from the form response of the front end and add it to the dictionary
    salary = request.form['salary']
    job_dict['Salary'] = salary
    # Get the other information from the form response of the front end and add it to the dictionary
    other_info_label = request.form['other_information_label'].capitalize()
    other_info = request.form['other_information']
    job_dict[other_info_label] = other_info
    # Append the newly created job to the data list
    data.append(job_dict)
    # Save the newly created job to the file directory
    save_data(job_dict)
    # Redirect back to the job display page
    return redirect('/index.html')

if __name__ == '__main__':
    app.run()