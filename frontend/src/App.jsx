import React, { useState, useEffect } from 'react'
import { recommend, feedback, stats, getUsers, getContents } from './api'

export default function App() {
  const [userId, setUserId] = useState('user_001')
  const [contentId, setContentId] = useState('content_001')
  const [users, setUsers] = useState([])
  const [contents, setContents] = useState([])
  const [result, setResult] = useState(null)
  const [impressionId, setImpressionId] = useState(null)
  const [statusMsg, setStatusMsg] = useState('')

  async function handleRecommend() {
    setStatusMsg('Requesting recommendation...')
    try {
      const res = await recommend(userId, contentId)
      setResult(res)
      setImpressionId(res.impression_id)
      setStatusMsg(`Got recommendation (latency ${Math.round(res.latency_ms)} ms)`)
    } catch (e) {
      setStatusMsg('Error: ' + (e.message || e))
    }
  }

  async function handleFeedback(reward = 1.0) {
    if (!impressionId) {
      setStatusMsg('No impression to give feedback for')
      return
    }
    setStatusMsg('Sending feedback...')
    try {
      const res = await feedback(impressionId, reward)
      setStatusMsg('Feedback recorded')
    } catch (e) {
      setStatusMsg('Error: ' + (e.message || e))
    }
  }

  async function handleStats() {
    setStatusMsg('Fetching stats...')
    try {
      const s = await stats()
      setStatusMsg(`CTR: ${s.overall_ctr} (${s.total_clicks}/${s.total_impressions})`)
    } catch (e) {
      setStatusMsg('Error: ' + (e.message || e))
    }
  }

  useEffect(() => {
    async function loadLists() {
      try {
        const [u, c] = await Promise.all([getUsers(), getContents()])
        setUsers(u || [])
        setContents(c || [])
        if (u && u.length && !userId) setUserId(u[0].user_id)
        if (c && c.length && !contentId) setContentId(c[0].content_id)
      } catch (e) {
        setStatusMsg('Failed to load users/contents: ' + (e.message || e))
      }
    }
    loadLists()
  }, [])

  return (
    <div className="container">
      <header>
        <h1>Artwork Bandit — Demo UI</h1>
        <p className="subtitle">Recommend artworks, send feedback, and view stats</p>
      </header>

      <main>
        <div className="controls">
          <label>
            User
            <select value={userId} onChange={(e) => setUserId(e.target.value)}>
              {users.map((u) => (
                <option key={u.user_id} value={u.user_id}>{u.user_id} {u.mood ? `— ${u.mood}` : ''}</option>
              ))}
            </select>
          </label>
          <label>
            Content
            <select value={contentId} onChange={(e) => setContentId(e.target.value)}>
              {contents.map((c) => (
                <option key={c.content_id} value={c.content_id}>{c.content_id} {c.title ? `— ${c.title}` : ''}</option>
              ))}
            </select>
          </label>
          <div className="actions">
            <button onClick={handleRecommend}>Recommend</button>
            <button onClick={() => handleFeedback(1.0)}>Click (reward=1)</button>
            <button onClick={() => handleFeedback(0.0)}>No Click (reward=0)</button>
            <button onClick={handleStats}>Stats</button>
          </div>
        </div>

        <section className="result">
          {result ? (
            <div className="card">
              <img src={result.artwork_image || ''} alt={result.artwork_id} />
              <div className="card-body">
                <h3>{result.artwork_id}</h3>
                <p>Algorithm: {result.algorithm}</p>
                <p>Latency: {Math.round(result.latency_ms)} ms</p>
                <p>Impression: {result.impression_id}</p>
              </div>
            </div>
          ) : (
            <p>No recommendation yet.</p>
          )}
        </section>

        <div className="status">{statusMsg}</div>
      </main>

      <footer>
        <small>Runs against API at <code>http://127.0.0.1:8000</code></small>
      </footer>
    </div>
  )
}
