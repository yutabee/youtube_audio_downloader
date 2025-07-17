# YouTube Audio Downloader

A high-performance YouTube audio downloader with advanced features including concurrent downloads, subtitle extraction, retry mechanisms, and rich progress display.

## Features

### Performance
- **Concurrent Downloads**: Download up to 8 playlist items simultaneously
- **Smart Retry**: Automatic retry mechanism for failed downloads
- **Rate Limiting**: Bandwidth control options

### User Experience
- **Rich UI**: Real-time progress bars with transfer speed and time remaining
- **Detailed Logging**: Comprehensive logging system for debugging
- **Statistics**: Download summary with success/fail counts

### Audio Features
- **Multiple Formats**: MP3, FLAC, WAV, M4A, Opus, Vorbis, AAC support
- **High Quality**: Custom bitrate settings
- **Metadata**: Automatic tagging and thumbnail embedding
- **Subtitle Extraction**: Convert YouTube subtitles to text files

### Reliability
- **Input Validation**: URL validation and error checking
- **Error Handling**: Individual error handling with detailed logging
- **Archive Management**: Prevent duplicate downloads

## Installation

```bash
# Install required packages
pip install yt-dlp rich

# Make executable (optional)
chmod +x youtube_audio_downloader.py
```

### Requirements
- **Python 3.6+**
- **yt-dlp**: YouTube video/audio extraction
- **rich**: Beautiful console UI
- **ffmpeg**: Audio conversion (must be installed on system)

## Basic Usage

```bash
# Download single video as MP3
./youtube_audio_downloader.py "https://youtu.be/VIDEO_ID"

# URLs with special characters must be quoted
./youtube_audio_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID&t=8s"

# Download with subtitles saved as text file
./youtube_audio_downloader.py "https://youtu.be/VIDEO_ID" -s

# Download with English subtitles
./youtube_audio_downloader.py "https://youtu.be/VIDEO_ID" -s --subtitle-lang en

# Download playlist in high-quality FLAC
./youtube_audio_downloader.py https://youtube.com/playlist?list=PLAYLIST_ID -f flac -b 320

# 8 concurrent downloads (faster for playlists)
./youtube_audio_downloader.py URL -w 8

# Custom output path
./youtube_audio_downloader.py URL -o "~/Music/YouTube/%(uploader)s/%(title)s.%(ext)s"
```

## Advanced Usage

### Download via Proxy
```bash
./youtube_audio_downloader.py URL --proxy socks5://localhost:9050
```

### Age-Restricted Videos (using cookies)
```bash
./youtube_audio_downloader.py URL --cookies cookies.txt
```

### Specific Playlist Items
```bash
./youtube_audio_downloader.py PLAYLIST_URL --playlist-items 1-3,5,7-10
```

### Rate Limiting and Archive
```bash
./youtube_audio_downloader.py URL --rate-limit 200K --archive downloaded.txt
```

## Command Line Options

### Basic Options
| Option | Description | Default |
|--------|-------------|---------|
| `-f, --format` | Audio format (mp3/flac/wav/m4a/opus/vorbis/aac) | mp3 |
| `-b, --bitrate` | Audio bitrate in kbps | 192 |
| `-o, --output` | Output filename template | %(title)s.%(ext)s |

### Download Control
| Option | Description | Default |
|--------|-------------|---------|
| `-w, --workers` | Number of concurrent downloads | 4 |
| `-r, --retry` | Number of retry attempts | 3 |
| `--retry-delay` | Delay between retries (seconds) | 5.0 |
| `--rate-limit` | Download rate limit (e.g., 50K, 4.2M) | None |

### Playlist Options
| Option | Description |
|--------|-------------|
| `--playlist-start` | Playlist start index |
| `--playlist-end` | Playlist end index |
| `--playlist-items` | Specific items (e.g., 1-3,5,7-10) |

### Subtitle Options
| Option | Description | Default |
|--------|-------------|---------|  
| `-s, --subtitles` | Download subtitles as text file | Disabled |
| `--subtitle-lang` | Subtitle language code | ja |

### Other Options
| Option | Description |
|--------|-------------|
| `--cookies` | Path to cookies file |
| `--proxy` | Proxy URL |
| `--archive` | Download archive file |
| `--no-metadata` | Disable metadata embedding |
| `--no-thumbnail` | Disable thumbnail embedding |
| `-v, --verbose` | Verbose output |
| `-q, --quiet` | Quiet mode |
| `--info` | Show detailed help |

## Output Filename Template

Uses yt-dlp format variables:

| Variable | Description |
|----------|-------------|
| `%(title)s` | Video title |
| `%(uploader)s` | Channel name |
| `%(upload_date)s` | Upload date |
| `%(duration)s` | Video duration |
| `%(view_count)s` | View count |
| `%(playlist)s` | Playlist name |
| `%(playlist_index)s` | Position in playlist |
| `%(ext)s` | File extension |

### Example Templates
```bash
# Organize by channel
-o "%(uploader)s/%(title)s.%(ext)s"

# Playlist with index
-o "%(playlist)s/%(playlist_index)02d - %(title)s.%(ext)s"

# Date-based organization
-o "%(upload_date)s/%(uploader)s - %(title)s.%(ext)s"
```

## Troubleshooting

### ffmpeg Not Found
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

### Region Blocked Content
Use proxy or VPN:
```bash
./youtube_audio_downloader.py URL --proxy socks5://proxy-server:port
```

### Age-Restricted Content
1. Export cookies from browser (using extension)
2. Use cookies file:
```bash
./youtube_audio_downloader.py URL --cookies cookies.txt
```

### Slow Downloads
Increase concurrent workers:
```bash
./youtube_audio_downloader.py URL -w 8  # Up to 8 concurrent
```

## Log Files

Detailed logs are saved to:
```
~/.youtube_audio_downloader/logs/download_YYYYMMDD_HHMMSS.log
```

## Important Notes

- Respect YouTube's Terms of Service
- Avoid downloading copyrighted content
- Use for personal purposes only
- Be mindful of server load

## License

This tool is created for educational purposes. Use at your own risk.

## Changelog

### v2.1.0 - Subtitle Support
- Added subtitle extraction to text files
- Support for both manual and auto-generated subtitles
- VTT to plain text conversion

### v2.0.0 - Major Update
- Added concurrent download support
- Rich progress display with detailed stats
- Comprehensive error handling
- Automatic metadata embedding
- Advanced logging system
- Proxy and cookie support