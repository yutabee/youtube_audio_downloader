#!/usr/bin/env python3
"""
改良版：音声ファイルからメロディを高精度で抽出するスクリプト
ボーカルメロディに特化した設定で、より正確な音符検出を実現
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import argparse
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional

# 音名の定義
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# A4 = 440Hz
A4_FREQ = 440.0
A4_MIDI = 69

# ボーカル音域の制限（より現実的な範囲）
VOCAL_FMIN = librosa.note_to_hz('C3')  # 約130Hz（男性の低音）
VOCAL_FMAX = librosa.note_to_hz('C6')  # 約1047Hz（女性の高音）


@dataclass
class Note:
    """音符を表すデータクラス"""
    pitch: float  # Hz
    midi_note: int
    note_name: str
    octave: int
    start_time: float
    duration: float
    confidence: float
    
    def __str__(self):
        return f"{self.note_name}{self.octave}"
    
    def with_details(self):
        return f"{self.note_name}{self.octave} ({self.pitch:.1f}Hz, {self.duration:.2f}s, conf:{self.confidence:.2f})"


class ImprovedMelodyExtractor:
    """改良版メロディ抽出クラス"""
    
    def __init__(self, audio_path: str, sr: int = 22050):
        self.audio_path = Path(audio_path)
        self.sr = sr
        self.y = None
        self.y_vocal = None
        self.notes = []
        self.tempo = None
        
    def load_audio(self):
        """音声ファイルを読み込む"""
        print(f"Loading audio file: {self.audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=self.sr)
        print(f"Audio loaded: {len(self.y)} samples at {self.sr} Hz")
        
    def separate_vocals(self):
        """ボーカルを分離（改良版）"""
        print("Separating vocal components...")
        
        # ハーモニック成分を抽出（より強力な設定）
        y_harmonic, y_percussive = librosa.effects.hpss(
            self.y, 
            margin=(1.0, 5.0),  # ハーモニック成分を優先
            kernel_size=31      # より大きなカーネルサイズ
        )
        
        # 中域を強調（ボーカル帯域）
        # バターワースフィルタでボーカル帯域を抽出
        from scipy import signal
        
        # バンドパスフィルタ設計（ボーカル帯域を広めに）
        nyquist = self.sr / 2
        low = 80 / nyquist     # 80Hz（男性ボーカルの下限）
        high = 2000 / nyquist  # 2000Hz（倍音を含む）
        
        # フィルタ次数を適度に
        b, a = signal.butter(4, [low, high], btype='band')
        
        # NaNチェック
        y_harmonic = np.nan_to_num(y_harmonic, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            self.y_vocal = signal.filtfilt(b, a, y_harmonic)
            # フィルタ後もNaNチェック
            self.y_vocal = np.nan_to_num(self.y_vocal, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            # フィルタが失敗した場合は元のデータを使用
            self.y_vocal = y_harmonic
        
        print("Vocal separation completed")
        
    def extract_pitches_pyin(self):
        """
        pYINアルゴリズムを使用してピッチを抽出（確率的YIN）
        """
        print("Extracting pitches using pYIN algorithm...")
        
        # フレーム設定
        frame_length = 2048
        hop_length = 512
        
        # pYINでピッチと信頼度を取得
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.y_vocal,
            fmin=VOCAL_FMIN,
            fmax=VOCAL_FMAX,
            sr=self.sr,
            frame_length=frame_length,
            hop_length=hop_length,
            fill_na=0
        )
        
        return f0, voiced_probs, hop_length
        
    def freq_to_note(self, freq: float) -> Tuple[str, int, int]:
        """周波数を音名とオクターブに変換"""
        if freq <= 0:
            return None, None, None
            
        # MIDIノート番号を計算
        midi_note = 12 * np.log2(freq / A4_FREQ) + A4_MIDI
        midi_note_rounded = int(np.round(midi_note))
        
        # 音名とオクターブを取得
        note_name = NOTE_NAMES[midi_note_rounded % 12]
        octave = (midi_note_rounded // 12) - 1
        
        return note_name, octave, midi_note_rounded
        
    def smooth_pitches_advanced(self, pitches: np.ndarray, confidences: np.ndarray, 
                               window_size: int = 7) -> np.ndarray:
        """
        高度なピッチスムージング（信頼度を考慮）
        """
        smoothed = np.zeros_like(pitches)
        
        for i in range(len(pitches)):
            # ウィンドウ内のインデックス
            start = max(0, i - window_size // 2)
            end = min(len(pitches), i + window_size // 2 + 1)
            
            # ウィンドウ内の有効なピッチ（信頼度が高いもの）
            window_pitches = pitches[start:end]
            window_confs = confidences[start:end]
            
            # 信頼度が0.5以上のピッチのみ使用
            valid_mask = (window_pitches > 0) & (window_confs > 0.5)
            
            if np.any(valid_mask):
                # 信頼度で重み付けした平均
                valid_pitches = window_pitches[valid_mask]
                valid_confs = window_confs[valid_mask]
                smoothed[i] = np.average(valid_pitches, weights=valid_confs)
            else:
                smoothed[i] = 0
                
        return smoothed
        
    def segment_notes_advanced(self, pitches: np.ndarray, confidences: np.ndarray, 
                              hop_length: int, min_duration: float = 0.08) -> List[Note]:
        """
        高度なノートセグメンテーション（信頼度を考慮）
        """
        notes = []
        current_note = None
        note_pitches = []
        note_confs = []
        start_frame = 0
        
        frame_to_time = hop_length / self.sr
        min_frames = int(min_duration / frame_to_time)
        
        for i, (pitch, conf) in enumerate(zip(pitches, confidences)):
            if pitch > 0 and conf > 0.3:  # 信頼度閾値
                note_info = self.freq_to_note(pitch)
                if note_info[0] is not None:
                    midi_note = note_info[2]
                    
                    # 新しいノートの検出（半音以上の変化）
                    if current_note is None or abs(midi_note - current_note) > 0.5:
                        # 前のノートを保存
                        if current_note is not None and len(note_pitches) >= min_frames:
                            # ノートの平均ピッチと信頼度を計算
                            avg_pitch = np.average(note_pitches, weights=note_confs)
                            avg_conf = np.mean(note_confs)
                            
                            final_note_info = self.freq_to_note(avg_pitch)
                            if final_note_info[0] is not None:
                                duration = len(note_pitches) * frame_to_time
                                notes.append(Note(
                                    pitch=avg_pitch,
                                    midi_note=final_note_info[2],
                                    note_name=final_note_info[0],
                                    octave=final_note_info[1],
                                    start_time=start_frame * frame_to_time,
                                    duration=duration,
                                    confidence=avg_conf
                                ))
                        
                        # 新しいノート開始
                        current_note = midi_note
                        note_pitches = [pitch]
                        note_confs = [conf]
                        start_frame = i
                    else:
                        # 同じノートの継続
                        note_pitches.append(pitch)
                        note_confs.append(conf)
            else:
                # 無音または低信頼度：ノートを終了
                if current_note is not None and len(note_pitches) >= min_frames:
                    avg_pitch = np.average(note_pitches, weights=note_confs)
                    avg_conf = np.mean(note_confs)
                    
                    final_note_info = self.freq_to_note(avg_pitch)
                    if final_note_info[0] is not None:
                        duration = len(note_pitches) * frame_to_time
                        notes.append(Note(
                            pitch=avg_pitch,
                            midi_note=final_note_info[2],
                            note_name=final_note_info[0],
                            octave=final_note_info[1],
                            start_time=start_frame * frame_to_time,
                            duration=duration,
                            confidence=avg_conf
                        ))
                
                current_note = None
                note_pitches = []
                note_confs = []
                
        return notes
        
    def filter_notes(self, notes: List[Note]) -> List[Note]:
        """ノートをフィルタリング（異常値を除去）"""
        if not notes:
            return notes
            
        # 音域でフィルタ（より厳密な範囲）
        filtered = []
        
        # 最初の数音は誤検出の可能性が高いので除外
        start_skip = min(3, len(notes) // 20)
        
        for i, note in enumerate(notes):
            # 最初の数音をスキップ
            if i < start_skip:
                continue
                
            # 極端な音域を除外（C#7などの明らかな誤検出を防ぐ）
            if note.octave < 2 or note.octave > 5:
                continue
                
            # C2のような低すぎる音も除外
            if note.midi_note < 48:  # C3未満
                continue
                
            # 極端に短い音符を除外
            if note.duration < 0.05:
                continue
                
            # 低信頼度を除外
            if note.confidence < 0.5:
                continue
                
            # 前後の音との差が極端な場合は除外（2.5オクターブ以上）
            if len(filtered) > 0:  # 既にフィルタされた音と比較
                prev_note = filtered[-1]
                if abs(note.midi_note - prev_note.midi_note) > 30:
                    continue
                
            filtered.append(note)
            
        return filtered
        
    def extract_melody(self):
        """メロディを抽出（メインメソッド）"""
        # 音声を読み込む
        self.load_audio()
        
        # ボーカルを分離
        self.separate_vocals()
        
        # テンポを検出
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        self.tempo = float(tempo) if hasattr(tempo, '__iter__') else tempo
        print(f"Detected tempo: {self.tempo:.1f} BPM")
        
        # ピッチ抽出
        pitches, confidences, hop_length = self.extract_pitches_pyin()
        
        # ピッチをスムージング
        pitches = self.smooth_pitches_advanced(pitches, confidences)
        
        # ノートセグメンテーション
        notes = self.segment_notes_advanced(pitches, confidences, hop_length)
        
        # フィルタリング
        self.notes = self.filter_notes(notes)
        
        print(f"Detected {len(self.notes)} valid notes")
        
        return self.notes
        
    def export_results(self, output_format: str = 'simple'):
        """結果を出力"""
        if not self.notes:
            print("No notes detected")
            return
            
        print(f"\n{'='*60}")
        print(f"IMPROVED MELODY EXTRACTION - {self.audio_path.name}")
        print(f"{'='*60}")
        print(f"Tempo: {self.tempo:.1f} BPM")
        print(f"Valid notes: {len(self.notes)}")
        
        # 音域の統計
        octaves = [n.octave for n in self.notes]
        print(f"Octave range: {min(octaves)} - {max(octaves)}")
        
        # 信頼度の統計
        confidences = [n.confidence for n in self.notes]
        print(f"Average confidence: {np.mean(confidences):.2f}")
        
        if output_format == 'simple':
            print("\nMelody sequence:")
            for i in range(0, len(self.notes), 12):
                note_str = " ".join([str(n) for n in self.notes[i:i+12]])
                print(note_str)
                
        elif output_format == 'detailed':
            print("\nDetailed notes (first 30):")
            print(f"{'Time':>6s} {'Note':>6s} {'Hz':>8s} {'Duration':>8s} {'Confidence':>10s}")
            print("-" * 50)
            
            for note in self.notes[:30]:
                print(f"{note.start_time:6.2f} {str(note):>6s} {note.pitch:8.1f} "
                      f"{note.duration:8.2f} {note.confidence:10.2f}")
                      
        # 音符の頻度分析
        note_counts = Counter([str(n) for n in self.notes])
        print(f"\nMost common notes:")
        for note, count in note_counts.most_common(8):
            percentage = (count / len(self.notes)) * 100
            print(f"  {note}: {count} times ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='改良版メロディ抽出ツール（ボーカル特化）'
    )
    
    parser.add_argument('audio_file', help='音声ファイルのパス')
    parser.add_argument('--format', choices=['simple', 'detailed'], 
                       default='simple', help='出力形式')
    
    args = parser.parse_args()
    
    if not Path(args.audio_file).exists():
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)
        
    # メロディ抽出
    extractor = ImprovedMelodyExtractor(args.audio_file)
    notes = extractor.extract_melody()
    extractor.export_results(output_format=args.format)


if __name__ == "__main__":
    main()