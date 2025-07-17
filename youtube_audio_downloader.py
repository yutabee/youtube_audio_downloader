#!/usr/bin/env python3
"""
YouTube Audio Downloader - Professional All-in-One Edition
----------------------------------------------------------
高性能な YouTube 音声ダウンローダー (単一ファイル版)

必要なパッケージのインストール:
    pip install yt-dlp rich

使用例:
    python youtube_audio_downloader.py https://youtu.be/VIDEO_ID
    python youtube_audio_downloader.py PLAYLIST_URL -f flac -b 320 -w 8
    python youtube_audio_downloader.py URL --help
"""
import argparse
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse

try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed. Please run: pip install yt-dlp")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        DownloadColumn,
        TransferSpeedColumn,
    )
    from rich.table import Table
    from rich.logging import RichHandler
except ImportError:
    print("Error: rich is not installed. Please run: pip install rich")
    sys.exit(1)


@dataclass
class DownloadConfig:
    """ダウンロード設定を管理するデータクラス"""
    audio_format: str
    output_template: str
    bitrate: str
    max_workers: int = 4
    retry_count: int = 3
    retry_delay: float = 5.0
    cookies_file: Optional[str] = None
    archive_file: Optional[str] = None
    metadata_embed: bool = True
    thumbnail_embed: bool = True
    quiet: bool = False
    verbose: bool = False
    rate_limit: Optional[str] = None
    proxy: Optional[str] = None
    geo_bypass: bool = True
    extract_flat: bool = False
    playlist_start: int = 1
    playlist_end: Optional[int] = None
    playlist_items: Optional[str] = None
    write_subtitles: bool = False
    subtitle_lang: str = 'ja'


class YouTubeAudioDownloader:
    """プロフェッショナル YouTube 音声ダウンローダー"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.console = Console()
        self.logger = self._setup_logging()
        self.download_stats = {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0
        }
        self._setup_directories()
    
    def _setup_logging(self) -> logging.Logger:
        """ロギングシステムのセットアップ"""
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        
        # ログディレクトリ作成
        log_dir = Path.home() / ".youtube_audio_downloader" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # ログファイル名（日付付き）
        log_file = log_dir / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # ロガー設定
        logger = logging.getLogger("YouTubeAudioDownloader")
        logger.setLevel(log_level)
        
        # ファイルハンドラー
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Rich コンソールハンドラー
        console_handler = RichHandler(console=self.console, rich_tracebacks=True)
        console_handler.setLevel(log_level)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_directories(self):
        """必要なディレクトリの作成"""
        output_path = Path(self.config.output_template).expanduser()
        output_dir = output_path.parent
        
        if output_dir != Path('.'):
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_dir}")
    
    def _validate_url(self, url: str) -> bool:
        """URL の妥当性チェック"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
            
            # YouTube URL パターンチェック
            valid_domains = [
                'youtube.com', 'www.youtube.com',
                'youtu.be', 'music.youtube.com',
                'm.youtube.com'
            ]
            return any(domain in result.netloc for domain in valid_domains)
        except Exception:
            return False
    
    def _build_ydl_opts(self, progress_callback=None) -> Dict[str, Any]:
        """yt-dlp オプション構築"""
        opts = {
            'format': 'bestaudio/best',
            'outtmpl': self.config.output_template,
            'ignoreerrors': False,  # エラーハンドリングは個別に行う
            'quiet': self.config.quiet,
            'no_warnings': self.config.quiet,
            'extract_flat': self.config.extract_flat,
            'geo_bypass': self.config.geo_bypass,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.config.audio_format,
                'preferredquality': self.config.bitrate,
            }],
            'writethumbnail': self.config.thumbnail_embed,
            'writeinfojson': False,
            'writedescription': False,
            'writesubtitles': self.config.write_subtitles,
            'writeautomaticsub': self.config.write_subtitles,
            'subtitleslangs': [self.config.subtitle_lang] if self.config.write_subtitles else [],
            'addmetadata': self.config.metadata_embed,
            'prefer_ffmpeg': True,
            'keepvideo': False,
            'postprocessor_args': {
                'FFmpegExtractAudio': [
                    '-acodec', self.config.audio_format,
                    '-b:a', f'{self.config.bitrate}k'
                ]
            }
        }
        
        # オプション設定
        if self.config.cookies_file:
            opts['cookiefile'] = self.config.cookies_file
        
        if self.config.archive_file:
            opts['download_archive'] = self.config.archive_file
        
        if self.config.rate_limit:
            opts['ratelimit'] = self._parse_rate_limit(self.config.rate_limit)
        
        if self.config.proxy:
            opts['proxy'] = self.config.proxy
        
        if self.config.playlist_items:
            opts['playlist_items'] = self.config.playlist_items
        else:
            opts['playliststart'] = self.config.playlist_start
            if self.config.playlist_end:
                opts['playlistend'] = self.config.playlist_end
        
        # メタデータ埋め込み設定
        if self.config.metadata_embed:
            opts['postprocessors'].append({
                'key': 'FFmpegMetadata',
                'add_metadata': True,
            })
        
        # サムネイル埋め込み設定
        if self.config.thumbnail_embed:
            opts['postprocessors'].append({
                'key': 'EmbedThumbnail',
                'already_have_thumbnail': False,
            })
        
        # プログレスフック
        if progress_callback:
            opts['progress_hooks'] = [progress_callback]
        
        return opts
    
    def _parse_rate_limit(self, rate_limit: str) -> int:
        """レート制限文字列をバイト数に変換"""
        rate_limit = rate_limit.upper()
        multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
        
        for suffix, multiplier in multipliers.items():
            if rate_limit.endswith(suffix):
                return int(float(rate_limit[:-1]) * multiplier)
        
        return int(rate_limit)
    
    def _extract_info(self, url: str) -> Optional[Dict]:
        """動画/プレイリスト情報の抽出"""
        try:
            opts = self._build_ydl_opts()
            opts['extract_flat'] = True
            opts['quiet'] = True
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as e:
            self.logger.error(f"Failed to extract info from {url}: {e}")
            return None
    
    def _process_subtitles(self, info: Dict, output_path: Path):
        """字幕をテキストファイルに変換して保存"""
        if not self.config.write_subtitles:
            return
        
        try:
            # 字幕情報の取得
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})
            
            # 優先順位: 手動字幕 > 自動字幕
            subtitle_data = None
            if self.config.subtitle_lang in subtitles:
                subtitle_data = subtitles[self.config.subtitle_lang]
                subtitle_type = "manual"
            elif self.config.subtitle_lang in automatic_captions:
                subtitle_data = automatic_captions[self.config.subtitle_lang]
                subtitle_type = "auto"
            else:
                # 言語が見つからない場合、利用可能な言語をログに出力
                available_langs = list(subtitles.keys()) + list(automatic_captions.keys())
                if available_langs:
                    self.logger.warning(
                        f"Subtitle language '{self.config.subtitle_lang}' not found. "
                        f"Available: {', '.join(set(available_langs))}"
                    )
                else:
                    self.logger.warning("No subtitles available for this video")
                return
            
            # VTT形式の字幕を探す
            vtt_url = None
            for sub in subtitle_data:
                if sub.get('ext') == 'vtt':
                    vtt_url = sub.get('url')
                    break
            
            if not vtt_url:
                self.logger.warning("No VTT subtitle format found")
                return
            
            # 字幕をダウンロード
            opts = {'quiet': True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                subtitle_content = ydl.urlopen(vtt_url).read().decode('utf-8')
            
            # VTT形式からプレーンテキストに変換
            text = self._vtt_to_text(subtitle_content)
            
            # テキストファイルとして保存
            text_path = output_path.with_suffix('.txt')
            text_path.write_text(text, encoding='utf-8')
            
            self.logger.info(
                f"Saved {subtitle_type} subtitles to: {text_path.name}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process subtitles: {e}")
    
    def _vtt_to_text(self, vtt_content: str) -> str:
        """VTT形式の字幕をプレーンテキストに変換"""
        # VTTヘッダーを削除
        lines = vtt_content.split('\n')
        text_lines = []
        
        # タイムスタンプとキューIDを除去
        timestamp_pattern = re.compile(r'^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}')
        cue_pattern = re.compile(r'^\d+$')
        
        for line in lines:
            line = line.strip()
            # 空行、タイムスタンプ、キューID、WEBVTTヘッダーをスキップ
            if (not line or 
                timestamp_pattern.match(line) or 
                cue_pattern.match(line) or 
                line.startswith('WEBVTT') or
                line.startswith('Kind:') or
                line.startswith('Language:')):
                continue
            
            # HTMLタグを除去
            line = re.sub(r'<[^>]+>', '', line)
            # 位置情報を除去
            line = re.sub(r'\{[^}]+\}', '', line)
            
            if line and line not in text_lines:  # 重複を避ける
                text_lines.append(line)
        
        return '\n'.join(text_lines)
    
    def _download_single(self, url: str, task_id=None, progress=None) -> Tuple[bool, str]:
        """単一動画のダウンロード"""
        attempt = 0
        last_error = ""
        
        while attempt < self.config.retry_count:
            try:
                if progress and task_id is not None:
                    progress.update(task_id, description=f"[cyan]Downloading: {url}")
                
                def progress_hook(d):
                    if progress and task_id is not None:
                        if d['status'] == 'downloading':
                            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                            downloaded = d.get('downloaded_bytes', 0)
                            if total > 0:
                                progress.update(task_id, completed=downloaded, total=total)
                        elif d['status'] == 'finished':
                            progress.update(task_id, description="[green]Processing audio...")
                
                opts = self._build_ydl_opts(progress_hook)
                
                # 字幕処理のために情報を取得
                info = None
                if self.config.write_subtitles:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
                
                # 字幕の処理
                if self.config.write_subtitles and info:
                    output_path = Path(self.config.output_template % {
                        'title': info.get('title', 'Unknown'),
                        'ext': self.config.audio_format,
                        'uploader': info.get('uploader', 'Unknown'),
                        'upload_date': info.get('upload_date', ''),
                        'id': info.get('id', ''),
                    })
                    self._process_subtitles(info, output_path)
                
                self.logger.info(f"Successfully downloaded: {url}")
                return True, ""
                
            except yt_dlp.utils.DownloadError as e:
                last_error = str(e)
                attempt += 1
                if attempt < self.config.retry_count:
                    self.logger.warning(
                        f"Download failed (attempt {attempt}/{self.config.retry_count}): {e}"
                    )
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error(f"Failed to download {url} after {self.config.retry_count} attempts: {e}")
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Unexpected error downloading {url}: {e}")
                break
        
        return False, last_error
    
    def _download_playlist_concurrent(self, playlist_info: Dict, progress: Progress):
        """プレイリストの並行ダウンロード"""
        entries = playlist_info.get('entries', [])
        if not entries:
            self.logger.warning("No entries found in playlist")
            return
        
        # プレイリストタスクの作成
        playlist_task = progress.add_task(
            f"[bold blue]Playlist: {playlist_info.get('title', 'Unknown')}",
            total=len(entries)
        )
        
        # ダウンロードタスクの準備
        download_tasks = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            for i, entry in enumerate(entries):
                if not entry:
                    continue
                
                video_url = entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}"
                video_title = entry.get('title', f'Video {i+1}')
                
                # 個別ダウンロードタスク作成
                task_id = progress.add_task(
                    f"[yellow]Queued: {video_title[:50]}...",
                    total=100
                )
                
                future = executor.submit(
                    self._download_single,
                    video_url,
                    task_id,
                    progress
                )
                download_tasks.append((future, video_title, task_id))
            
            # ダウンロード結果の処理
            completed = 0
            for future, title, task_id in download_tasks:
                try:
                    success, error = future.result()
                    completed += 1
                    
                    if success:
                        self.download_stats['success'] += 1
                        progress.update(
                            task_id,
                            description=f"[green]✓ {title[:50]}...",
                            completed=100
                        )
                    else:
                        self.download_stats['failed'] += 1
                        progress.update(
                            task_id,
                            description=f"[red]✗ {title[:50]}... ({error})"
                        )
                    
                    progress.update(playlist_task, completed=completed)
                    
                except Exception as e:
                    self.logger.error(f"Task failed: {e}")
                    self.download_stats['failed'] += 1
    
    def download(self, urls: List[str]):
        """メインダウンロード処理"""
        self.logger.info(f"Starting download of {len(urls)} URL(s)")
        
        # URL バリデーション
        valid_urls = []
        for url in urls:
            if self._validate_url(url):
                valid_urls.append(url)
            else:
                self.logger.error(f"Invalid YouTube URL: {url}")
                self.download_stats['failed'] += 1
        
        if not valid_urls:
            self.console.print("[red]No valid URLs provided")
            return
        
        # プログレスバー設定
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=self.console
        ) as progress:
            
            for url in valid_urls:
                # 情報抽出
                info = self._extract_info(url)
                if not info:
                    self.download_stats['failed'] += 1
                    continue
                
                # プレイリストか単体動画かを判定
                if info.get('_type') == 'playlist':
                    self.logger.info(f"Processing playlist: {info.get('title', 'Unknown')}")
                    self._download_playlist_concurrent(info, progress)
                else:
                    # 単体動画
                    task_id = progress.add_task(
                        f"[yellow]{info.get('title', 'Unknown')[:50]}...",
                        total=100
                    )
                    success, error = self._download_single(url, task_id, progress)
                    
                    if success:
                        self.download_stats['success'] += 1
                        progress.update(
                            task_id,
                            description=f"[green]✓ {info.get('title', 'Unknown')[:50]}...",
                            completed=100
                        )
                    else:
                        self.download_stats['failed'] += 1
                        progress.update(
                            task_id,
                            description=f"[red]✗ {info.get('title', 'Unknown')[:50]}... ({error})"
                        )
        
        self._print_summary()
    
    def _print_summary(self):
        """ダウンロード結果サマリーの表示"""
        table = Table(title="Download Summary", show_header=True)
        table.add_column("Status", style="cyan", no_wrap=True)
        table.add_column("Count", justify="right")
        
        table.add_row("✓ Success", f"[green]{self.download_stats['success']}")
        table.add_row("✗ Failed", f"[red]{self.download_stats['failed']}")
        table.add_row("⊘ Skipped", f"[yellow]{self.download_stats['skipped']}")
        table.add_row(
            "[bold]Total",
            f"[bold]{sum(self.download_stats.values())}"
        )
        
        self.console.print("\n")
        self.console.print(table)
        self.console.print("\n")
        
        # ログファイルの場所を表示
        log_location = Path.home() / ".youtube_audio_downloader" / "logs"
        self.console.print(f"[dim]Logs saved to: {log_location}[/dim]")


def print_help():
    """ヘルプメッセージの表示"""
    help_text = """
YouTube Audio Downloader - Professional Edition
==============================================

主な機能:
- 並行ダウンロード対応（プレイリスト高速処理）
- 自動リトライ機構
- リッチなプログレスバー表示
- メタデータ自動タグ付け
- 字幕の文字起こし機能
- 詳細ログ記録

基本的な使い方:
    python youtube_audio_downloader.py https://youtu.be/VIDEO_ID
    python youtube_audio_downloader.py URL -s  # 字幕も保存
    python youtube_audio_downloader.py URL -s --subtitle-lang en  # 英語字幕
    python youtube_audio_downloader.py PLAYLIST_URL -f flac -b 320
    python youtube_audio_downloader.py URL -w 8  # 8並行ダウンロード

高度な使い方:
    # プロキシ経由
    python youtube_audio_downloader.py URL --proxy socks5://localhost:9050
    
    # クッキー使用（年齢制限動画など）
    python youtube_audio_downloader.py URL --cookies cookies.txt
    
    # プレイリストの特定範囲
    python youtube_audio_downloader.py PLAYLIST --playlist-items 1-3,5,7-10

出力ファイル名のカスタマイズ:
    -o "%(uploader)s/%(title)s.%(ext)s"  # アップロード者別フォルダ
    -o "%(playlist)s/%(playlist_index)02d - %(title)s.%(ext)s"  # プレイリスト整理

利用可能な変数:
    %(title)s        - 動画タイトル
    %(uploader)s     - アップロード者
    %(upload_date)s  - アップロード日
    %(duration)s     - 動画の長さ
    %(view_count)s   - 再生回数
    %(playlist)s     - プレイリスト名
    %(playlist_index)s - プレイリスト内番号

トラブルシューティング:
    1. ffmpeg エラー → ffmpeg をインストール
       macOS: brew install ffmpeg
       Ubuntu: sudo apt-get install ffmpeg
    
    2. 地域制限 → --proxy オプションを使用
    
    3. 年齢制限 → --cookies オプションを使用

詳細オプション: --help
"""
    console = Console()
    console.print(help_text, style="bright_blue")


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Professional YouTube Audio Downloader with advanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single video as MP3
  %(prog)s https://youtu.be/VIDEO_ID
  
  # Download with subtitles (Japanese)
  %(prog)s https://youtu.be/VIDEO_ID -s
  
  # Download with English subtitles
  %(prog)s https://youtu.be/VIDEO_ID -s --subtitle-lang en
  
  # Download playlist with custom settings
  %(prog)s https://youtube.com/playlist?list=PLAYLIST_ID -f flac -b 320 -w 8
  
  # Download with cookies and proxy
  %(prog)s URL --cookies cookies.txt --proxy socks5://localhost:9050
  
  # Download specific playlist items
  %(prog)s PLAYLIST_URL --playlist-items 1-3,5,7-10
        """
    )
    
    # 必須引数
    parser.add_argument("urls", nargs="*", help="YouTube video or playlist URL(s)")
    
    # 音声フォーマット設定
    format_group = parser.add_argument_group("Audio Format Options")
    format_group.add_argument(
        "-f", "--format",
        default="mp3",
        choices=["mp3", "flac", "wav", "m4a", "opus", "vorbis", "aac"],
        help="Audio format (default: mp3)"
    )
    format_group.add_argument(
        "-b", "--bitrate",
        default="192",
        help="Audio bitrate in kbps (default: 192)"
    )
    
    # 出力設定
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output",
        default="%(title)s.%(ext)s",
        help="Output filename template (yt-dlp format)"
    )
    output_group.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't embed metadata in audio files"
    )
    output_group.add_argument(
        "--no-thumbnail",
        action="store_true",
        help="Don't embed thumbnail in audio files"
    )
    
    # 字幕設定
    subtitle_group = parser.add_argument_group("Subtitle Options")
    subtitle_group.add_argument(
        "-s", "--subtitles",
        action="store_true",
        help="Download and save subtitles as text file"
    )
    subtitle_group.add_argument(
        "--subtitle-lang",
        default="ja",
        help="Subtitle language code (default: ja)"
    )
    
    # ダウンロード設定
    download_group = parser.add_argument_group("Download Options")
    download_group.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of concurrent downloads (default: 4)"
    )
    download_group.add_argument(
        "-r", "--retry",
        type=int,
        default=3,
        help="Number of retry attempts (default: 3)"
    )
    download_group.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Delay between retries in seconds (default: 5.0)"
    )
    download_group.add_argument(
        "--rate-limit",
        help="Download rate limit (e.g., 50K, 4.2M)"
    )
    download_group.add_argument(
        "--archive",
        help="Download archive file to track downloaded videos"
    )
    
    # プレイリスト設定
    playlist_group = parser.add_argument_group("Playlist Options")
    playlist_group.add_argument(
        "--playlist-start",
        type=int,
        default=1,
        help="Playlist start index (default: 1)"
    )
    playlist_group.add_argument(
        "--playlist-end",
        type=int,
        help="Playlist end index"
    )
    playlist_group.add_argument(
        "--playlist-items",
        help="Specific playlist items to download (e.g., 1-3,5,7-10)"
    )
    
    # 認証・プロキシ設定
    auth_group = parser.add_argument_group("Authentication & Proxy")
    auth_group.add_argument(
        "--cookies",
        help="Path to cookies file"
    )
    auth_group.add_argument(
        "--proxy",
        help="Proxy URL (e.g., socks5://localhost:9050)"
    )
    auth_group.add_argument(
        "--geo-bypass",
        action="store_true",
        default=True,
        help="Bypass geographic restrictions (default: enabled)"
    )
    
    # ロギング設定
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - suppress output"
    )
    logging_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output for debugging"
    )
    
    # その他
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed help and examples"
    )
    
    args = parser.parse_args()
    
    # 詳細ヘルプ表示
    if args.info:
        print_help()
        sys.exit(0)
    
    # URL が指定されていない場合
    if not args.urls:
        parser.print_help()
        sys.exit(1)
    
    # 設定オブジェクト作成
    config = DownloadConfig(
        audio_format=args.format,
        output_template=args.output,
        bitrate=args.bitrate,
        max_workers=args.workers,
        retry_count=args.retry,
        retry_delay=args.retry_delay,
        cookies_file=args.cookies,
        archive_file=args.archive,
        metadata_embed=not args.no_metadata,
        thumbnail_embed=not args.no_thumbnail,
        quiet=args.quiet,
        verbose=args.verbose,
        rate_limit=args.rate_limit,
        proxy=args.proxy,
        geo_bypass=args.geo_bypass,
        playlist_start=args.playlist_start,
        playlist_end=args.playlist_end,
        playlist_items=args.playlist_items,
        write_subtitles=args.subtitles,
        subtitle_lang=args.subtitle_lang
    )
    
    # ダウンローダー実行
    try:
        downloader = YouTubeAudioDownloader(config)
        downloader.download(args.urls)
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[yellow]Download interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console = Console()
        console.print(f"\n[red]Fatal error: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()