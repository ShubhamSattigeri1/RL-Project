import axios from 'axios'

const BASE = 'https://sattigeri07-rl-project.hf.space'

export async function recommend(user_id, content_id) {
  const r = await axios.post(`${BASE}/recommend`, { user_id, content_id })
  return r.data
}

export async function feedback(impression_id, reward) {
  const r = await axios.post(`${BASE}/feedback`, { impression_id, reward })
  return r.data
}

export async function stats() {
  const r = await axios.get(`${BASE}/stats`)
  return r.data
}

export async function getUsers() {
  const r = await axios.get(`${BASE}/users`)
  return r.data
}

export async function getContents() {
  const r = await axios.get(`${BASE}/contents`)
  return r.data
}
