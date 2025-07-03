# JIRA Integration Setup Guide

## Issue Fixed
The JIRA submission was not actually saving stories to JIRA because the backend was only simulating the submission. This has been fixed with proper JIRA API integration.

## Recent Updates (Fixed Issues)

### 1. User Story Description Issue - FIXED ✅
**Problem**: User story descriptions were showing as generic fallback text instead of proper descriptions.

**Root Cause**: The user story generator agent only creates `name` field, but frontend was looking for `description` field.

**Fix**: Updated backend to:
- Map `name` field to `title` for consistency
- Generate proper user story descriptions in "As a user, I want..." format
- Handle both `description` and `summary` fields

### 2. Responsible Systems Issue - FIXED ✅
**Problem**: JIRA submissions always showed "CAPS, CMS" instead of actual responsible systems.

**Root Cause**: Frontend was hardcoded to send "CAPS, CMS" instead of using actual systems from user story.

**Fix**: Updated both frontend and backend to:
- Use actual `systems` array from user story generator
- Convert systems array to comma-separated string format
- Include responsible systems in JIRA description
- Fallback to "CAPS, CMS" only if no systems specified

### 3. Priority Field Issue - FIXED ✅
**Problem**: JIRA was returning error "Field 'priority' cannot be set. It is not on the appropriate screen, or unknown."

**Root Cause**: The priority field is not available or configured in the JIRA project screen configuration.

**Fix**: Updated backend to:
- Make priority field optional in JIRA story creation
- Attempt story creation without priority if priority field fails
- Provide user-friendly error messages for common JIRA configuration issues
- Show warnings when story is created successfully but some fields were skipped

## What Was Changed

1. **Backend Integration**: Modified `poc2_backend_processor_three_section.py` to:
   - Import JIRA and dotenv libraries
   - Add JIRA client initialization 
   - Create `create_jira_story()` helper function
   - Update `three_section_submit_jira()` to actually call JIRA API

2. **Frontend Updates**: Modified `poc2_three_section_layout.html` to:
   - Display proper success/error messages
   - Show JIRA ticket link when successful
   - Handle API errors gracefully

3. **JIRA Connector**: Enhanced `jira_connectorv2.py` to:
   - Add story creation functionality (previously only had Epic creation)

## Setup Instructions

### 1. Get JIRA API Token
1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Give it a label (e.g., "Three Section App")
4. Copy the generated token

### 2. Create Environment File
1. Copy `.env.example` to `.env`
2. Fill in your JIRA API token:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   JIRA_API_TOKEN=your_jira_api_token_here
   ```

### 3. Verify JIRA Permissions
Make sure your JIRA user (lalluluke@gmail.com) has:
- Permission to create issues in the SCRUM project
- Access to the Story issue type

### 4. Test the Integration
1. Start the three-section backend: `python poc2_backend_processor_three_section.py`
2. Go through the normal flow to create a user story
3. Click "Submit to Jira"
4. If successful, you'll see the JIRA ticket ID and a link to view it in JIRA

### 5. Verify the Fixes
**Check User Story Descriptions:**
- User stories should now show proper descriptions instead of generic text
- Descriptions should be in "As a user, I want..." format

**Check Responsible Systems:**
- JIRA tickets should show actual system names (e.g., "Customer acquisition platform, Credit decision engine") 
- No more hardcoded "CAPS, CMS" unless no systems are specified
- Systems should appear in the JIRA description under "Responsible Systems:"

## Troubleshooting

### Common Issues:

1. **"JIRA client not available"**: 
   - Check that JIRA_API_TOKEN is set in .env file
   - Verify the API token is valid

2. **Permission errors**:
   - Verify your JIRA user has Create Issue permission in SCRUM project
   - Check if Story issue type is available in the project

3. **Field errors (FIXED)**:
   - ~~Priority field errors~~ - Now handled automatically
   - The Epic Link field (customfield_10014) might be different in your setup
   - Check JIRA admin settings for custom field IDs

4. **Priority field not available**:
   - This is now handled automatically - stories will be created without priority if the field is not available
   - You'll see a warning message but the story will still be created successfully

### Logs
Check the application logs for detailed error messages:
- Backend logs will show JIRA API responses
- Browser console will show frontend errors

## Features Added

- ✅ Actual JIRA API integration (no more simulation)
- ✅ Real ticket creation with proper formatting
- ✅ Acceptance criteria included in description
- ✅ Error handling and user feedback
- ✅ Direct links to created JIRA tickets
- ✅ Proper success/failure messaging

## Next Steps

If you encounter issues:
1. Check the backend logs for JIRA API errors
2. Verify your .env file configuration
3. Test with the standalone JIRA connector: `python jira_connectorv2.py`
