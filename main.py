# Required imports
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define a structure for tracking job applications
class JobApplication(BaseModel):
    title: str
    company: str
    status: str  # Applied/Skipped/Error
    reason: str  # Reason for skip or error if applicable

class ApplicationResults(BaseModel):
    applications: List[JobApplication]

# Create a controller with Chrome profile settings to use existing login
controller = Controller(
    output_model=ApplicationResults,
    browser="chrome",  # Changed from browser_config
    use_existing_profile=True  # Moved out of browser_config
)

async def main():
    # Initialize the LLM (using API key from .env)
    llm = ChatOpenAI(model="gpt-4o")

    # Define the specific task instructions for the agent
    task = """
    1. Go to indeed.com
    2. Search for "Solutions Engineer" jobs in San Francisco
    3. Apply these filters:
       - Date posted: Last 3 days
       - Distance: Within 25 miles
    4. For each job listing that has an "Easily Apply" button:
       - Click to apply
       - Use the resume already stored in Indeed
       - Complete any additional application steps
       - Track the application status
    5. Keep track of which jobs were applied to and any issues encountered
    """

    # Create and run the agent with the specified configuration
    agent = Agent(
        task=task,
        llm=llm,
        controller=controller
    )

    # Execute the agent and get the history
    history = await agent.run()

    # Process and display the results
    result = history.final_result()
    if result:
        applications = ApplicationResults.model_validate_json(result)
        
        # Print a summary of all applications
        print("\nApplication Summary:")
        for app in applications.applications:
            print(f"\nPosition: {app.title}")
            print(f"Company: {app.company}")
            print(f"Status: {app.status}")
            if app.reason:
                print(f"Notes: {app.reason}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())