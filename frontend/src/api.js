import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

export const fetchSummary = async () => {
  console.log('[RNIA] Fetching summary from', `${API_URL}/summary`);
  const response = await axios.get(`${API_URL}/summary`);
  console.log('[RNIA] Summary response:', response.data);
  return response.data;
};

export const fetchNews = async () => {
  console.log('[RNIA] Fetching news from', `${API_URL}/news`);
  const response = await axios.get(`${API_URL}/news`);
  console.log('[RNIA] News response: received', response.data.length, 'articles');
  return response.data;
};

export const analyzeArticle = async (text) => {
  console.log('[RNIA] Analyzing article, text length:', text.length);
  const response = await axios.post(`${API_URL}/analyze`, { text });
  console.log('[RNIA] Analyze response:', response.data);
  return response.data;
};
