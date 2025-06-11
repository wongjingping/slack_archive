#!/usr/bin/env python3
"""
Slack Message Thread Extractor

This script extracts your Slack messages with threads, reactions, and AI summaries.
Requires Slack API token and OpenAI API key.
"""

import os
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import pandas as pd
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import openai
from pydantic import BaseModel
from tqdm import tqdm


class MessageSummary(BaseModel):
    """Structured output model for message summary and educational score."""

    summary: str
    educational_score: int


class SlackMessageExtractor:
    def __init__(self, slack_token: str, openai_api_key: str):
        """Initialize the extractor with API credentials."""
        self.slack_client = WebClient(token=slack_token)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.user_id = None
        self.rate_limit_mean = 0.05  # Mean for exponential distribution

    def get_user_id(self) -> str:
        """Get the current user's ID."""
        if self.user_id:
            return self.user_id

        try:
            response = self.slack_client.auth_test()
            self.user_id = response["user_id"]
            print(f"Authenticated as user ID: {self.user_id}")
            return self.user_id
        except SlackApiError as e:
            raise Exception(f"Failed to authenticate: {e.response['error']}")

    def get_channels(self) -> List[Dict]:
        """Get list of channels the user is a member of."""
        channels = []
        try:
            # Try to get both public and private channels
            response = self.slack_client.conversations_list(
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=1000,
            )
            channels.extend(response["channels"])
            tqdm.write(f"Found {len(channels)} channels (public + private)")

        except SlackApiError as e:
            if e.response["error"] == "missing_scope" and "groups:read" in str(
                e.response.get("needed", "")
            ):
                # Fallback to public channels only if missing groups:read scope
                tqdm.write(
                    "Warning: Missing 'groups:read' scope, falling back to public channels only"
                )
                try:
                    response = self.slack_client.conversations_list(
                        types="public_channel", exclude_archived=True, limit=1000
                    )
                    channels.extend(response["channels"])
                    tqdm.write(f"Found {len(channels)} public channels")
                except SlackApiError as e2:
                    tqdm.write(
                        f"Error fetching public channels: {e2.response['error']}"
                    )
                    return []
            else:
                tqdm.write(f"Error fetching channels: {e.response['error']}")
                return []

        return channels

    def get_messages_from_channel(
        self,
        channel_id: str,
        channel_name: str,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
    ) -> List[Dict]:
        """Get messages from a specific channel, optionally filtered by date range."""
        messages = []
        cursor = None
        user_id = self.get_user_id()

        try:
            with tqdm(desc=f"Fetching from #{channel_name}", unit=" batches") as pbar:
                while True:
                    # Build API call parameters
                    api_params = {"channel": channel_id, "limit": 200, "cursor": cursor}

                    # Add timestamp filters if provided
                    if start_timestamp:
                        api_params["oldest"] = str(start_timestamp)
                    if end_timestamp:
                        api_params["latest"] = str(end_timestamp)

                    response = self.slack_client.conversations_history(**api_params)

                    batch_messages = response.get("messages", [])
                    # Filter for messages sent by the user
                    user_messages = [
                        msg for msg in batch_messages if msg.get("user") == user_id
                    ]
                    messages.extend(user_messages)
                    pbar.update(1)

                    if not response.get("has_more", False):
                        break

                    cursor = response.get("response_metadata", {}).get("next_cursor")
                    time.sleep(
                        np.random.exponential(self.rate_limit_mean)
                    )  # Exponential jitter

        except SlackApiError as e:
            tqdm.write(
                f"Error fetching messages from {channel_name}: {e.response['error']}"
            )

        tqdm.write(f"Found {len(messages)} messages from you in #{channel_name}")
        return messages

    def get_thread_replies(self, channel_id: str, thread_ts: str) -> List[Dict]:
        """Get all replies in a thread, filtered for user's messages."""
        user_id = self.get_user_id()

        try:
            response = self.slack_client.conversations_replies(
                channel=channel_id, ts=thread_ts, limit=1000
            )

            # Filter for user's messages only, excluding the original message
            replies = [
                msg
                for msg in response.get("messages", [])[1:]
                if msg.get("user") == user_id
            ]

            time.sleep(np.random.exponential(self.rate_limit_mean))
            return replies

        except SlackApiError as e:
            if e.response["error"] == "rate_limited":
                # If rate limited, wait and retry once
                retry_after = int(e.response.get("headers", {}).get("Retry-After", 1))
                tqdm.write(f"Rate limited, waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.get_thread_replies(channel_id, thread_ts)
            else:
                # Silently ignore errors to reduce noise
                pass
            return []

    def get_reactions(self, channel_id: str, timestamp: str) -> Dict[str, int]:
        """Get emoji reactions for a specific message."""
        try:
            response = self.slack_client.reactions_get(
                channel=channel_id, timestamp=timestamp
            )

            reactions = {}
            if "message" in response and "reactions" in response["message"]:
                for reaction in response["message"]["reactions"]:
                    emoji = reaction["name"]
                    count = reaction["count"]
                    reactions[emoji] = count

            # Small delay to avoid overwhelming the API
            time.sleep(np.random.exponential(self.rate_limit_mean))
            return reactions

        except SlackApiError as e:
            if e.response["error"] == "rate_limited":
                # If rate limited, wait and retry once
                retry_after = int(e.response.get("headers", {}).get("Retry-After", 1))
                tqdm.write(f"Rate limited, waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.get_reactions(channel_id, timestamp)
            elif e.response["error"] != "no_reaction":
                # Only print error if it's not just "no reactions"
                pass  # Silently ignore to reduce noise
            return {}

    def extract_attachments(self, message: Dict) -> List[str]:
        """Extract attachment names from a message."""
        attachment_names = []

        # Check for files
        if "files" in message:
            for file_obj in message["files"]:
                name = file_obj.get("name") or file_obj.get("title") or "unnamed_file"
                attachment_names.append(name)

        # Check for attachments (links, etc.)
        if "attachments" in message:
            for attachment in message["attachments"]:
                if "title" in attachment:
                    attachment_names.append(attachment["title"])
                elif "fallback" in attachment:
                    attachment_names.append(attachment["fallback"])

        return attachment_names

    def summarize_with_gpt(self, text: str, message_id: str = "") -> Tuple[str, int]:
        """Use GPT-4o-mini to summarize text and rate educational value using structured output."""
        try:
            prompt = f"""Analyze this Slack message/thread and provide:
1. A summary in exactly 7 words or fewer
2. An educational/learning value score from 1-5 (where 1 is casual chat / administrative updates, 3 contains something generally useful / progress updates, 5 is highly nuanced and educational learning about a specific topic)

# Examples

Content:
"Won't be able to join the team meeting today"
Expected Summary: "Administrative"
Expected Educational score: 1

Content:
"Sorry for the late message! This week for me was spent
• Adding feature A
• Fixed bug B
• Speeded up feature C"
Expected Summary: "Progress update"
Expected Educational score: 2

Content:
"You can install the remote ssh extension in VSCode and after connecting to our remote GPU instances, you can use VSCode to edit files on the remote machine as you normally would locally."
Expected Summary: "VSCode remote ssh extension"
Expected Educational score: 3

Content:
"Hi all sharing a little pytorch gradient clipping titbit that I learnt today,
gradient clipping only affects the `.grad` tensor in of the tensor, not the actual tensor values itself! Here's an example:
```import math
import torch

t1 = torch.tensor([-1.0, math.sqrt(2.0), 1.0])
max_norm = torch.nn.utils.clip_grad_norm_(t1, 1.0)
print(max_norm) # tensor(0.). huh?
print(t1) # same old t1 tensor([-1.0000,  1.4142,  1.0000]). why isn't it working?```
Reason is because all tensors' `.grad` defaults to 0 and so no gradient was present and no gradient was clipped. This is the proper way to test out gradient clipping - by setting the gradient first:
```t1.grad = torch.tensor([1.0, -math.sqrt(2.0), 1.0])
max_norm = torch.nn.utils.clip_grad_norm_(t1, 1.0, norm_type=2)
print(max_norm)
# tensor(2.)
print(t1.grad, t1)
# tensor([ 0.5000, -0.7071,  0.5000]) tensor([-1.0000,  1.4142,  1.0000])```
Notice that the original t1 hasn't been modified - only it's `.grad` attribute has been"
Expected Summary: "Pytorch gradient clipping"
Expected Educational score: 5

Message/Thread:
`
{text}
`
"""

            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format=MessageSummary,
                max_tokens=100,
                temperature=0.0,
            )

            # Extract structured data
            parsed = response.choices[0].message.parsed
            summary = parsed.summary
            score = max(1, min(5, parsed.educational_score))  # Ensure 1-5 range

            return summary, score

        except Exception as e:
            tqdm.write(f"Error with OpenAI API for message {message_id}: {e}")
            return "Error generating summary", 5

    def summarize_batch_parallel(
        self, message_texts: List[Tuple[str, str]], max_workers: int = 20
    ) -> Dict[str, Tuple[str, int]]:
        """Process multiple texts in parallel using ThreadPoolExecutor."""
        results = {}

        if not message_texts:
            return results

        def process_single_message(text_and_id):
            text, msg_id = text_and_id
            return msg_id, self.summarize_with_gpt(text, msg_id)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(process_single_message, text_and_id): text_and_id[1]
                for text_and_id in message_texts
            }

            # Collect results as they complete with progress bar
            with tqdm(
                total=len(message_texts), desc="OpenAI summaries", unit=" messages"
            ) as pbar:
                for future in as_completed(future_to_id):
                    try:
                        msg_id, (summary, score) = future.result()
                        results[msg_id] = (summary, score)
                        pbar.update(1)
                    except Exception as e:
                        msg_id = future_to_id[future]
                        tqdm.write(f"Error processing message {msg_id}: {e}")
                        results[msg_id] = ("Error generating summary", 5)
                        pbar.update(1)

        return results

    def process_messages(
        self,
        channels: List[str] = None,
        max_workers: int = 20,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """Process all messages and extract thread data."""
        message_data = []

        # Get channels to process
        if channels:
            # Convert channel names to IDs if needed
            all_channels = self.get_channels()
            channel_map = {ch["name"]: ch for ch in all_channels}
            channels_to_process = []

            for ch in channels:
                if ch.startswith("#"):
                    ch = ch[1:]  # Remove # prefix
                if ch in channel_map:
                    channels_to_process.append(channel_map[ch])
                else:
                    tqdm.write(f"Warning: Channel '{ch}' not found")
        else:
            channels_to_process = self.get_channels()

        # Parse date range if provided
        start_timestamp = None
        end_timestamp = None
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                start_timestamp = start_dt.timestamp()
                tqdm.write(f"Filtering messages from: {start_date}")
            except ValueError:
                tqdm.write(
                    f"Warning: Invalid start date format '{start_date}'. Expected YYYY-MM-DD"
                )

        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                # Add 24 hours to include the entire end date
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
                end_timestamp = end_dt.timestamp()
                tqdm.write(f"Filtering messages until: {end_date}")
            except ValueError:
                tqdm.write(
                    f"Warning: Invalid end date format '{end_date}'. Expected YYYY-MM-DD"
                )

        # First pass: collect basic message data (fast)
        raw_messages = []

        for channel in channels_to_process:
            channel_id = channel["id"]
            channel_name = channel["name"]

            messages = self.get_messages_from_channel(
                channel_id, channel_name, start_timestamp, end_timestamp
            )

            for message in messages:
                # Skip if message is itself a thread reply
                if "thread_ts" in message and message["ts"] != message["thread_ts"]:
                    continue

                msg_timestamp = float(message["ts"])
                msg_datetime = datetime.fromtimestamp(msg_timestamp)
                raw_messages.append(
                    {
                        "message": message,
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "msg_ts": message["ts"],
                        "datetime": msg_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        if not raw_messages:
            tqdm.write("No messages found to process")
            return []

        # Second pass: fetch metadata in parallel
        def process_message_metadata(raw_msg):
            try:
                message = raw_msg["message"]
                channel_id = raw_msg["channel_id"]
                channel_name = raw_msg["channel_name"]
                msg_ts = raw_msg["msg_ts"]

                msg_text = message.get("text", "")

                # Get thread replies if this message has a thread
                thread_messages = []
                if "thread_ts" in message or "reply_count" in message:
                    thread_replies = self.get_thread_replies(channel_id, msg_ts)
                    thread_messages = [
                        reply.get("text", "") for reply in thread_replies
                    ]

                # Get reactions
                reactions = self.get_reactions(channel_id, msg_ts)

                # Get attachments
                attachments = self.extract_attachments(message)

                # Combine all text for summarization
                full_text = msg_text
                if thread_messages:
                    full_text += "\n\nThread replies:\n" + "\n".join(thread_messages)

                return {
                    "datetime": raw_msg["datetime"],
                    "msg_1": msg_text,
                    "msg_thread": (
                        " | ".join(thread_messages) if thread_messages else ""
                    ),
                    "files": ", ".join(attachments) if attachments else "",
                    "reactions": json.dumps(reactions) if reactions else "",
                    "channel": channel_name,
                    "full_text": full_text.strip(),
                    "msg_ts": msg_ts,
                }
            except Exception as e:
                print(
                    f"Error processing message {raw_msg.get('msg_ts', 'unknown')}: {e}"
                )
                return None

        # Process metadata in parallel with ThreadPoolExecutor
        metadata_workers = min(
            10, len(raw_messages)
        )  # Limit concurrent Slack API calls
        message_data = []

        with ThreadPoolExecutor(max_workers=metadata_workers) as executor:
            future_to_msg = {
                executor.submit(process_message_metadata, raw_msg): raw_msg
                for raw_msg in raw_messages
            }

            with tqdm(
                total=len(raw_messages), desc="Fetching metadata", unit=" messages"
            ) as pbar:
                for future in as_completed(future_to_msg):
                    result = future.result()
                    if result:
                        message_data.append(result)
                    pbar.update(1)

        if not message_data:
            tqdm.write("No valid message data after processing")
            return []

        # Prepare texts for parallel summarization
        texts_to_summarize = []
        for msg_info in message_data:
            if msg_info["full_text"]:
                texts_to_summarize.append((msg_info["full_text"], msg_info["msg_ts"]))

        # Second pass: parallel summarization
        summaries = {}
        if texts_to_summarize:
            summaries = self.summarize_batch_parallel(texts_to_summarize, max_workers)

        # Third pass: combine everything into final results
        results = []
        for msg_info in message_data:
            msg_ts = msg_info["msg_ts"]

            if msg_info["full_text"] and msg_ts in summaries:
                summary, edu_score = summaries[msg_ts]
            else:
                summary = "Empty message"
                edu_score = 1

            result = {
                "datetime": msg_info["datetime"],
                "channel": msg_info["channel"],
                "msg_1": msg_info["msg_1"],
                "msg_thread": msg_info["msg_thread"],
                "files": msg_info["files"],
                "summary": summary,
                "educational_score": edu_score,
                "reactions": msg_info["reactions"],
            }

            results.append(result)

        tqdm.write(f"Processed {len(results)} total messages")
        return results

    def save_to_csv(self, results: List[Dict], filename: str = "slack_messages.csv"):
        """Save results to CSV file."""
        if not results:
            tqdm.write("No results to save")
            return

        # Prepare data for CSV (excluding extra fields)
        csv_data = []
        for result in results:
            csv_row = {
                "datetime": result["datetime"],
                "channel": result["channel"],
                "msg_1": result["msg_1"],
                "msg_thread": result["msg_thread"],
                "files": result["files"],
                "summary": result["summary"],
                "educational_score": result["educational_score"],
                "reactions": result["reactions"],
            }
            csv_data.append(csv_row)

        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        tqdm.write(f"Saved {len(results)} messages to {filename}")


def main():
    """Main function to run the extractor."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract your Slack messages with threads, reactions, and AI summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slack_message_extractor.py                    # Process all public channels
  python slack_message_extractor.py general random     # Process only #general and #random
  python slack_message_extractor.py -c engineering -c support  # Process #engineering and #support
  python slack_message_extractor.py --start-date 2024-05-01 --end-date 2024-06-01  # Filter by date range
  python slack_message_extractor.py general --start-date 2024-05-01  # May 1st onwards in #general
        """,
    )
    parser.add_argument(
        "channels",
        nargs="*",
        help="Channel names to process (without # prefix). If not provided, all public channels will be processed.",
    )
    parser.add_argument(
        "-c",
        "--channel",
        action="append",
        dest="additional_channels",
        help="Additional channel to process (can be used multiple times)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="slack_messages.csv",
        help="Output CSV filename (default: slack_messages.csv)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=20,
        help="Number of parallel workers for OpenAI API calls (default: 20)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for message filtering (YYYY-MM-DD format, e.g., 2024-05-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for message filtering (YYYY-MM-DD format, e.g., 2024-06-01)",
    )

    args = parser.parse_args()

    # Combine channels from positional args and -c flags
    channels_to_process = args.channels or []
    if args.additional_channels:
        channels_to_process.extend(args.additional_channels)

    # Remove duplicates while preserving order
    if channels_to_process:
        channels_to_process = list(dict.fromkeys(channels_to_process))
        tqdm.write(
            f"Processing specific channels: {', '.join(f'#{ch}' for ch in channels_to_process)}"
        )
    else:
        channels_to_process = None
        tqdm.write("Processing all public channels")

    load_dotenv()

    # Get API keys from environment variables
    slack_token = os.getenv("SLACK_USER_TOKEN") or os.getenv("SLACK_BOT_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not slack_token:
        tqdm.write(
            "Error: SLACK_BOT_TOKEN or SLACK_USER_TOKEN not found in environment variables"
        )
        tqdm.write(
            "Please add your Slack token to a .env file or set as environment variable"
        )
        return

    if not openai_key:
        tqdm.write("Error: OPENAI_API_KEY not found in environment variables")
        tqdm.write(
            "Please add your OpenAI API key to a .env file or set as environment variable"
        )
        return

    # Initialize extractor
    extractor = SlackMessageExtractor(slack_token, openai_key)

    # Process messages
    tqdm.write("Starting message extraction...")
    results = extractor.process_messages(
        channels_to_process, args.workers, args.start_date, args.end_date
    )

    # Save to CSV
    extractor.save_to_csv(results, args.output)

    tqdm.write("Extraction complete!")


if __name__ == "__main__":
    main()
