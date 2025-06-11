# Slack Message Thread Extractor

This script extracts your Slack messages with threads, reactions, and generates a 7-word summary + educational score (1-5). It processes messages where you are the thread starter and includes all your subsequent messages in those threads.

## Getting Started

### Prerequisites
- **Slack API Access**: User (preferred) or bot token. If using a bot token, you need to add the bot to every channel you want to process, which can be a bit of a hassle.
- **OpenAI API Key**: For content summarization and scoring of educational value.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Slack API Token

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Give it a name and select your workspace
4. Go to "OAuth & Permissions" and add these Bot Token Scopes:
   - `channels:history` - View messages and other content in a user's public channels
   - `channels:read` - View basic information about public channels in a workspace
   - `groups:read` - View private channels in a workspace
   - `links:read` - View URLs in messages
   - `reactions:read` - View emoji reactions in a user's channels and conversations and their associated content
   - `users:read` - View people in a workspace
5. Install the app to your workspace
6. Copy the "Bot User OAuth Token" (starts with `xoxb-`)

### 3. Get OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Create a new API key and copy it

### 4. Configure Environment Variables

1. Copy `env_template.txt` to `.env`:
   ```bash
   cp env_template.txt .env
   ```

2. Edit `.env` and add your tokens:
   ```
   SLACK_USER_TOKEN=xoxb-your-actual-token-here
   # or if using a bot token, you need to add the bot to every channel you want to process
   SLACK_BOT_TOKEN=xoxb-your-actual-token-here
   OPENAI_API_KEY=sk-your-actual-openai-key-here
   ```

## Usage

### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `channels` | Channel names to process (positional) | `general random` |
| `-c, --channel` | Specify channels (can be used multiple times) | `-c general -c random` |
| `-o, --output` | Custom output CSV filename | `-o my_messages.csv` |
| `-w, --workers` | Number of parallel workers for OpenAI API calls | `-w 10` |
| `--start-date` | Filter messages from this date onwards (YYYY-MM-DD) | `--start-date 2024-05-01` |
| `--end-date` | Filter messages until this date (YYYY-MM-DD) | `--end-date 2024-06-01` |

### Sample Command

Extract messages from #general, #random and #engineering channels, filter by date range, and save to `messages.csv` with 15 parallel workers.
```bash
python slack_message_extractor.py general random -c engineering -o messages.csv -w 15 --start-date 2024-05-01 --end-date 2024-06-01
```

## How It Works

The script authenticates with Slack and OpenAI, fetches specified channels (or all public channels), processes messages to find threads you started, gets reactions and attachments, generates AI summaries, and saves results to CSV.

## Output Format

The script creates a CSV file with these columns:

| Column | Description |
|--------|-------------|
| `datetime` | When the first message was sent (YYYY-MM-DD HH:MM:SS) |
| `channel` | Name of the Slack channel where the message was posted |
| `msg_1` | Your first message that started the thread |
| `msg_thread` | All your subsequent messages in the thread (separated by " \| ") |
| `files` | Attachment filenames (comma-separated) |
| `summary` | GPT-4o-mini generated summary (max 7 words) |
| `educational_score` | Educational value score from 1-5 (1=casual chat, 5=highly educational) |
| `reactions` | JSON string of emoji reactions ({"emoji": count}) |

## Performance & Rate Limiting

- **OpenAI API calls**: Uses ThreadPoolExecutor with 20 parallel workers by default (customizable with `-w`)
- **Slack API**: Built-in rate limiting with exponential backoff and automatic retry
- **Progress tracking**: Shows completion progress for all operations

## Customization

### Modify AI Prompts and Scoring

You can easily customize the educational value scoring and summary generation by modifying the prompt in the `summarize_with_gpt()` method (around line 227 in `slack_message_extractor.py`). The current prompt focuses on technical/learning content, but you could adapt it for:

- Different scoring criteria (e.g., sentiment analysis, urgency, project relevance)
- Alternative summary styles (e.g., keyword extraction, humor, sentiment)
- Custom scoring ranges or categories

### Add Custom Output Fields

To extract additional semantic metadata from your messages, you can:

1. **Extend the `MessageSummary` model** (line 27) to include new fields
2. **Update the prompt** to request the additional information
3. **Modify the CSV output** in `save_to_csv()` method (line 528) to include your new fields

Example: Add sentiment analysis, humor, or technical complexity scores.

### Performance Tuning

#### Speed Up Slack API Calls
If you want faster Slack API queries, you can reduce the rate limiting delay by modifying `self.rate_limit_mean` (line 39). The default is 0.05 seconds between requests:

```python
self.rate_limit_mean = 0.02  # Faster queries (use responsibly)
```

#### Optimize OpenAI Calls
Adjust the number of parallel workers with the `-w` flag based on your OpenAI rate limits:

```bash
python slack_message_extractor.py -w 30  # More parallel requests
```

### Performance Benchmarks

For reference, extracting 222 messages from a single channel took:
- **18 seconds** for Slack API queries (can be reduced with faster rate limiting)
- **11 seconds** for OpenAI API processing (scales with parallel workers)

> **Note**: Be respectful of API rate limits. Slack's API is generally more tolerant than others, but always monitor for rate limit errors in the output.
