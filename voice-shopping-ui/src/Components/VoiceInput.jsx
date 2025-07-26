import React, { useState, useRef } from "react";
import styles from "./VoiceInput.module.css";
import { transcribeAudio } from "../api/assistant";

const VoiceInput = ({ onTranscribe }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState("");
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await processAudio(audioBlob);
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setTranscript("");
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error accessing microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processAudio = async (audioBlob) => {
    setIsProcessing(true);
    try {
      const audioFile = new File([audioBlob], "recording.wav", { type: "audio/wav" });
      const result = await transcribeAudio(audioFile);
      
      if (result.transcription) {
        setTranscript(result.transcription);
        onTranscribe(result.transcription);
      } else {
        alert('No speech detected. Please try again.');
      }
    } catch (error) {
      console.error('Error transcribing audio:', error);
      alert('Error processing audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetTranscript = () => {
    setTranscript("");
    if (isRecording) {
      stopRecording();
    }
  };

  return (
    <div className={styles.voiceInputContainer}>
      <div className={styles.voiceControls}>
        <button 
          onClick={startRecording} 
          disabled={isRecording || isProcessing}
          className={`${styles.button} ${isRecording ? styles.recording : ''}`}
        >
          {isRecording ? '🔴 Recording...' : '🎙 Start Voice'}
        </button>
        
        <button 
          onClick={stopRecording} 
          disabled={!isRecording || isProcessing}
          className={styles.button}
        >
          🛑 Stop & Send
        </button>
        
        <button 
          onClick={resetTranscript} 
          className={styles.resetButton}
          disabled={isProcessing}
        >
          {isProcessing ? '⏳ Processing...' : '🔄 Reset'}
        </button>
      </div>

      {transcript && (
        <div className={styles.transcriptBox}>
          <div className={styles.transcriptLabel}>Last transcript:</div>
          <div className={styles.transcriptText}>{transcript}</div>
        </div>
      )}

      {isProcessing && (
        <div className={styles.processingIndicator}>
          <div className={styles.spinner}></div>
          <span>Processing your voice...</span>
        </div>
      )}
    </div>
  );
};

export default VoiceInput;