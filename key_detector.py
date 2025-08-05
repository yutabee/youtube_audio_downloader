#!/usr/bin/env python3
"""
音声ファイルからメロディのキーを高精度に分析するスクリプト

必要なライブラリ:
    pip install librosa numpy scipy matplotlib
    
オプション（より高精度な分析のため）:
    pip install essentia
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import sys
from pathlib import Path

# キーとコード進行のマッピング
KEY_PROFILES = {
    # メジャーキー（Krumhansl-Kessler profiles）
    'C': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    'C#': [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],
    'D': [2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66],
    'D#': [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],
    'E': [2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19],
    'F': [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],
    'F#': [2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09],
    'G': [4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38],
    'G#': [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33],
    'A': [2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48],
    'A#': [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23],
    'B': [2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35],
    
    # マイナーキー
    'Cm': [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    'C#m': [3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34],
    'Dm': [3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69],
    'D#m': [2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98],
    'Em': [3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75],
    'Fm': [4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54],
    'F#m': [2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53],
    'Gm': [3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60],
    'G#m': [2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38],
    'Am': [5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52],
    'A#m': [3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68],
    'Bm': [2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33],
}

# 音名のマッピング
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ENHARMONIC_NAMES = {
    'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb',
    'C#m': 'Dbm', 'D#m': 'Ebm', 'F#m': 'Gbm', 'G#m': 'Abm', 'A#m': 'Bbm'
}


class KeyDetector:
    """高精度キー検出クラス"""
    
    def __init__(self, audio_path: str, sr: int = 22050):
        """
        初期化
        
        Args:
            audio_path: 音声ファイルのパス
            sr: サンプリングレート
        """
        self.audio_path = Path(audio_path)
        self.sr = sr
        self.y = None
        self.chromagram = None
        self.tempo = None
        self.key_scores = {}
        
    def load_audio(self):
        """音声ファイルを読み込む"""
        print(f"Loading audio file: {self.audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr)
        print(f"Audio loaded: {len(self.y)} samples at {self.sr} Hz")
        
    def extract_chromagram(self, hop_length: int = 512):
        """
        クロマグラムを抽出（ピッチクラス分布）
        
        Args:
            hop_length: ホップ長
        """
        print("Extracting chromagram...")
        
        # ハーモニック成分を抽出（より正確なピッチ検出のため）
        y_harmonic, _ = librosa.effects.hpss(self.y)
        
        # 複数の手法でクロマグラムを計算し、平均を取る
        chroma_stft = librosa.feature.chroma_stft(y=y_harmonic, sr=self.sr, hop_length=hop_length)
        chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr, hop_length=hop_length)
        chroma_cens = librosa.feature.chroma_cens(y=y_harmonic, sr=self.sr, hop_length=hop_length)
        
        # 重み付き平均（CQTベースの手法により高い重みを与える）
        self.chromagram = 0.2 * chroma_stft + 0.5 * chroma_cqt + 0.3 * chroma_cens
        
        # 正規化
        self.chromagram = librosa.util.normalize(self.chromagram, axis=0)
        
        print(f"Chromagram shape: {self.chromagram.shape}")
        
    def extract_tempo(self):
        """テンポを抽出（オプション）"""
        print("Extracting tempo...")
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.tempo = float(tempo)
        print(f"Estimated tempo: {self.tempo:.2f} BPM")
        
    def calculate_key_correlations(self):
        """各キーとの相関を計算"""
        print("Calculating key correlations...")
        
        # クロマグラムの時間平均を計算
        chroma_mean = np.mean(self.chromagram, axis=1)
        
        # 各キープロファイルとの相関を計算
        for key, profile in KEY_PROFILES.items():
            # ピアソン相関係数
            correlation = np.corrcoef(chroma_mean, profile)[0, 1]
            self.key_scores[key] = correlation
            
    def detect_key_segments(self, segment_length: float = 10.0):
        """
        曲をセグメントに分割してキーを検出（転調検出）
        
        Args:
            segment_length: セグメント長（秒）
        """
        print(f"Detecting key changes in {segment_length}s segments...")
        
        hop_length = 512
        segment_frames = int(segment_length * self.sr / hop_length)
        n_segments = self.chromagram.shape[1] // segment_frames
        
        segment_keys = []
        
        for i in range(n_segments):
            start = i * segment_frames
            end = min((i + 1) * segment_frames, self.chromagram.shape[1])
            segment_chroma = self.chromagram[:, start:end]
            
            # セグメントの平均クロマ
            segment_mean = np.mean(segment_chroma, axis=1)
            
            # 各キーとの相関
            segment_scores = {}
            for key, profile in KEY_PROFILES.items():
                correlation = np.corrcoef(segment_mean, profile)[0, 1]
                segment_scores[key] = correlation
            
            # 最高スコアのキー
            best_key = max(segment_scores, key=segment_scores.get)
            segment_keys.append(best_key)
            
        return segment_keys
    
    def get_alternative_notation(self, key: str) -> str:
        """異名同音の表記を取得"""
        return ENHARMONIC_NAMES.get(key, key)
    
    def analyze(self, show_plot: bool = True):
        """
        完全な分析を実行
        
        Args:
            show_plot: グラフを表示するか
        """
        # 音声を読み込む
        self.load_audio()
        
        # テンポを抽出
        self.extract_tempo()
        
        # クロマグラムを抽出
        self.extract_chromagram()
        
        # キー相関を計算
        self.calculate_key_correlations()
        
        # 結果をソート
        sorted_keys = sorted(self.key_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 最も可能性の高いキー
        best_key = sorted_keys[0][0]
        best_score = sorted_keys[0][1]
        
        # 次に可能性の高いキー
        second_key = sorted_keys[1][0]
        second_score = sorted_keys[1][1]
        
        # 転調をチェック
        segment_keys = self.detect_key_segments()
        key_counts = Counter(segment_keys)
        most_common_keys = key_counts.most_common(3)
        
        # 結果を表示
        print("\n" + "="*50)
        print("KEY DETECTION RESULTS")
        print("="*50)
        print(f"\n分析ファイル: {self.audio_path.name}")
        print(f"テンポ: {self.tempo:.1f} BPM")
        print(f"\n最も可能性の高いキー: {best_key} (信頼度: {best_score:.3f})")
        
        # 異名同音表記
        if best_key in ENHARMONIC_NAMES:
            print(f"異名同音表記: {self.get_alternative_notation(best_key)}")
        
        print(f"\n次に可能性の高いキー: {second_key} (信頼度: {second_score:.3f})")
        
        # 相対的なキー（平行調）
        if 'm' in best_key:
            relative_major = NOTE_NAMES[(NOTE_NAMES.index(best_key.replace('m', '')) + 3) % 12]
            print(f"\n平行長調: {relative_major}")
        else:
            relative_minor = NOTE_NAMES[(NOTE_NAMES.index(best_key) - 3) % 12] + 'm'
            print(f"\n平行短調: {relative_minor}")
        
        # 転調の可能性
        if len(most_common_keys) > 1 and most_common_keys[1][1] > len(segment_keys) * 0.2:
            print("\n転調の可能性:")
            for key, count in most_common_keys[:3]:
                percentage = (count / len(segment_keys)) * 100
                print(f"  {key}: {percentage:.1f}% of the song")
        
        # 上位5つのキー候補
        print("\n上位5つのキー候補:")
        for i, (key, score) in enumerate(sorted_keys[:5]):
            alt = f" ({self.get_alternative_notation(key)})" if key in ENHARMONIC_NAMES else ""
            print(f"  {i+1}. {key}{alt}: {score:.3f}")
        
        # グラフを表示
        if show_plot:
            self.plot_analysis(best_key)
            
        return best_key, best_score
    
    def plot_analysis(self, detected_key: str):
        """分析結果をプロット"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. クロマグラム
        librosa.display.specshow(
            self.chromagram,
            y_axis='chroma',
            x_axis='time',
            sr=self.sr,
            hop_length=512,
            ax=axes[0]
        )
        axes[0].set_title(f'Chromagram - {self.audio_path.name}')
        axes[0].set_ylabel('Pitch Class')
        
        # 2. 平均クロマベクトル
        chroma_mean = np.mean(self.chromagram, axis=1)
        axes[1].bar(NOTE_NAMES, chroma_mean)
        axes[1].set_title('Average Pitch Class Distribution')
        axes[1].set_xlabel('Pitch Class')
        axes[1].set_ylabel('Magnitude')
        axes[1].grid(True, alpha=0.3)
        
        # 3. キースコア
        keys = list(self.key_scores.keys())
        scores = list(self.key_scores.values())
        
        # メジャーとマイナーで色分け
        colors = ['blue' if 'm' not in k else 'red' for k in keys]
        
        axes[2].bar(range(len(keys)), scores, color=colors, alpha=0.7)
        axes[2].set_xticks(range(len(keys)))
        axes[2].set_xticklabels(keys, rotation=45, ha='right')
        axes[2].set_title(f'Key Correlation Scores (Detected: {detected_key})')
        axes[2].set_xlabel('Key')
        axes[2].set_ylabel('Correlation')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 凡例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Major'),
            Patch(facecolor='red', alpha=0.7, label='Minor')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'key_analysis_{self.audio_path.stem}.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='高精度音楽キー検出ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用
  python key_detector.py "audio/Mrs. GREEN APPLE - 僕のこと.mp3"
  
  # グラフを表示しない
  python key_detector.py "audio/song.mp3" --no-plot
  
  # 高いサンプリングレートで分析
  python key_detector.py "audio/song.mp3" --sr 44100
        """
    )
    
    parser.add_argument('audio_file', help='分析する音声ファイルのパス')
    parser.add_argument('--sr', type=int, default=22050, help='サンプリングレート (default: 22050)')
    parser.add_argument('--no-plot', action='store_true', help='グラフを表示しない')
    
    args = parser.parse_args()
    
    # ファイルの存在確認
    if not Path(args.audio_file).exists():
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)
    
    # キー検出を実行
    detector = KeyDetector(args.audio_file, sr=args.sr)
    key, confidence = detector.analyze(show_plot=not args.no_plot)
    
    print(f"\n最終結果: {key} (信頼度: {confidence:.3f})")


if __name__ == "__main__":
    main()