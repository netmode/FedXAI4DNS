# Dataset Dashboard

This dashboard is a web application built using [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/) to provide an interactive interface for analyzing domain name characteristics. The application allows users to explore different metrics related to domain name features, including character analysis, length and ratio distributions, entropy and complexity, and more.

## Features

- **Overview**: Displays a pie chart showing the distribution of different domain families.
- **Character Analysis**: Analyzes the presence of individual characters in domain names.
- **Length and Ratio Metrics**: Shows histograms for the length distribution of domain names.
- **Entropy and Complexity**: Provides insights into the entropy of domain names and their complexity.
- **Domain Name Analysis**: Plots the relationship between the longest sequence of letters and entropy in domain names.
- **Domain Attribute Viewer**: An interactive interface to view domain-specific attributes.

## Requirements

- Python 3.7 or higher
- Dash
- Dash Bootstrap Components
- Plotly
- Pandas


## Getting Started

1. **Prepare the Dataset**: The application expects a CSV file named `dataset.csv` with relevant domain data.

2. **Run the Dashboard**: 
   Open the terminal, navigate to the project directory, and run the app using the following command:

3. **Access the Dashboard**:
   After starting the server, open a web browser and go to http://127.0.0.1:8050/ to access the dashboard.

## Usage

Navigate through different sections of the dashboard:

- **Select filters** in the sidebar to focus on specific domain families.
- **Switch between dark and light modes** using the toggle in the main view.
- **Interact with graphs** for each domain characteristic to gain insights.

## Code Structure

- **Sidebar and Layout**: The sidebar navigation enables access to various sections.
- **Callbacks**: The app uses Dash callbacks to load different views based on URL and user input.
- **Client-Side Theme Switch**: The app supports switching between light and dark themes.

