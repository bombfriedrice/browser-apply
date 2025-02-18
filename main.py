# Required imports
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig, Controller
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file (just for OpenAI key)
load_dotenv()

# Define a structure for tracking job applications
class JobApplication(BaseModel):
    title: str
    company: str
    status: str  # Applied/Skipped/Error
    reason: str  # Reason for skip or error if applicable

class ApplicationResults(BaseModel):
    applications: List[JobApplication]

# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)

# Create a controller with the configured browser
controller = Controller(output_model=ApplicationResults)

async def main():
    # Initialize the LLM (using API key from .env)
    llm = ChatOpenAI(
        model="gpt-4o",
    )

    # Define the specific task instructions for the agent
    task = """
    1. Go to indeed.com wait a random amount of seconds from 0.1 - 3 seconds per action to bypass any human verification from cloudflare.
    2. Search for "now hiring" jobs in San Francisco and click search button.
    3. If you get a human verify message select you are human.  
    4. Then, Apply these filters:
       - Date posted: Last 3 days
       - Distance: Within 25 miles
    4. For each job listing that has an "Easily Apply" tag:
       - Click to apply
       - Make sure Resume Resume_RJ2.pdf is the one selected and click 'continue' if you get confused just click 'continue' do not try to exit and save. we want to complete the application.
       - If it takes you to a window that says Select job experience that is relevant, just click 'continue'
       - When prompted for personal information, use:
         * Name: RJ Moscardon
         * Email: rjmoscardon@gmail.com
         * Phone: 5105704027
         * Location: San Francisco, CA
         * LinkedIn: linkedin.com/in/rjmoscardon
       - Complete any additional application steps and click 'continue' for voluntary information select do not want to share
       - Track the application status
    5. Keep track of which jobs were applied to and any issues encountered
    """
    
    
    # Create and run the agent with the specified configuration
    agent = Agent(
        task=task,
        llm=llm,
        controller=controller,
        browser=browser
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

    # Wait for user input before closing the browser
    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())