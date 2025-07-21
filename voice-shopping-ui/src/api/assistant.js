import axios from 'axios';

const BASE_URL = 'http://localhost:5050'; // Update for production

export const sendTextToAssistant = async (text) => {
  const response = await axios.post(`${BASE_URL}/chat`, { text });
  return response.data;
};

export const transcribeAudio = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post(`${BASE_URL}/transcribe`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const resetSession = async () => {
  await axios.post(`${BASE_URL}/reset`);
};
