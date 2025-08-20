import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";
import "./App.css";

const API_URL = process.env.SQLITE_FASTAPI_URL; 
const USER_ID_KEY = "chat_user_id";
const SESSIONS_KEY = "chat_sessions";
const CONVS_KEY = "chat_conversations";

export default function App() {
  const [userId, setUserId] = useState(null);
  const [sessions, setSessions] = useState([]); // array of session ids
  const [activeSession, setActiveSession] = useState(null);
  const [conversations, setConversations] = useState({}); 
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [ensuring, setEnsuring] = useState(false);
  const messagesEndRef = useRef(null);

  // helpers to read/save localStorage 
  function loadLocalState() {
    const uid = localStorage.getItem(USER_ID_KEY) || null;
    const savedSessions = JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]");
    const savedConvs = JSON.parse(localStorage.getItem(CONVS_KEY) || "{}");
    return { uid, savedSessions, savedConvs };
  }
  function saveSessionsToLocal(sessions) {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  }
  function saveConversationsToLocal(convs) {
    localStorage.setItem(CONVS_KEY, JSON.stringify(convs));
  }

  
  useEffect(() => {
    (async () => {
      // user id
      let uid = localStorage.getItem(USER_ID_KEY);
      if (!uid) {
        uid = uuidv4();
        localStorage.setItem(USER_ID_KEY, uid);
      }
      setUserId(uid);

      // load sessions + conversations from localStorage
      const { savedSessions, savedConvs } = loadLocalState();
      setConversations(savedConvs || {});
      if (savedSessions && savedSessions.length > 0) {
        setSessions(savedSessions);
        // Ensure + fetch history for the first session before selecting it
        const first = savedSessions[0];
        await ensureAndLoad(uid, first, { selectAfterLoad: true, showCachedFirst: true });
      } else {
        // no sessions stored then create first session and ensure it
        const sid = `session_${Date.now()}`;
        setSessions([sid]);
        setConversations((p) => ({ ...p, [sid]: [] }));
        saveSessionsToLocal([sid]);
        saveConversationsToLocal({ ...(savedConvs || {}), [sid]: [] });
        await ensureAndLoad(uid, sid, { selectAfterLoad: true, showCachedFirst: false });
      }
    })();
  }, []);

  // scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations, activeSession, loading]);

  // server/session helpers 
  async function ensureSessionOnServer(uid, sessionId) {
    setEnsuring(true);
    try {
      await axios.post(`${API_URL}/sessions/ensure`, {
        user_id: uid,
        session_id: sessionId,
      });
      return true;
    } catch (err) {
      console.error("ensureSession error:", err?.response?.data || err.message);
      return false;
    } finally {
      setEnsuring(false);
    }
  }

  async function fetchHistoryFromServer(uid, sessionId) {
    try {
      const res = await axios.get(
        `${API_URL}/history/${encodeURIComponent(uid)}/${encodeURIComponent(sessionId)}`
      );
      const msgs = Array.isArray(res.data.messages) ? res.data.messages : [];
      return msgs;
    } catch (err) {
      // server might return [] or error then return null to signal a fetch error
      console.error("fetchHistory error:", err?.response?.data || err.message);
      return null;
    }
  }

  // ensure session and then load history, update local cache and optionally select it
  async function ensureAndLoad(uid, sessionId, opts = { selectAfterLoad: true, showCachedFirst: true }) {
    // If showCachedFirst true, set activeSession to sessionId immediately so UI shows cached messages quickly
    if (opts.showCachedFirst) {
      setActiveSession(sessionId);
    }

    // Ensure exists server-side
    await ensureSessionOnServer(uid, sessionId);

    // Fetch history from server
    const serverMsgs = await fetchHistoryFromServer(uid, sessionId);

    setConversations((prev) => {
      const cached = Array.isArray(prev[sessionId]) ? prev[sessionId] : [];
      let finalMsgs;
      if (Array.isArray(serverMsgs)) {
        // if server has messages, use them; if server returned empty, keep cached
        finalMsgs = serverMsgs.length > 0 ? serverMsgs : cached;
      } else {
        // server failed then keep cached (or empty)
        finalMsgs = cached;
      }
      const next = { ...prev, [sessionId]: finalMsgs };
      saveConversationsToLocal(next);
      return next;
    });

    if (opts.selectAfterLoad) {
      setActiveSession(sessionId);
    }
  }

  // create a new session and ensure+load it
  async function createAndSelectSession(uid, sessionId) {
    // add session locally and persist
    setSessions((prev) => {
      if (prev.includes(sessionId)) return prev;
      const next = [sessionId, ...prev];
      saveSessionsToLocal(next);
      return next;
    });

    // add empty conv locally for immediate UI
    setConversations((prev) => {
      const next = { ...prev };
      if (!next[sessionId]) next[sessionId] = [];
      saveConversationsToLocal(next);
      return next;
    });

    // ensure and load from server (this will setActiveSession)
    await ensureAndLoad(uid, sessionId, { selectAfterLoad: true, showCachedFirst: true });
  }

  // when user clicks a session -> show cached messages immediately, then ensure+fetch from server and replace/merge
  async function handleSelectSession(sessionId) {
    // show cached messages quickly and then fetch/merge from server
    await ensureAndLoad(userId, sessionId, { selectAfterLoad: true, showCachedFirst: true });
  }

  // new chat
  async function handleNewChat() {
    const newId = `session_${Date.now()}`;
    await createAndSelectSession(userId, newId);
  }

  // delete session locally (no server delete)
  function handleDeleteSession(id) {
    setSessions((prev) => {
      const next = prev.filter((s) => s !== id);
      saveSessionsToLocal(next);
      return next;
    });
    setConversations((prev) => {
      const copy = { ...prev };
      delete copy[id];
      saveConversationsToLocal(copy);
      return copy;
    });
    if (activeSession === id) {
      const remaining = sessions.filter((s) => s !== id);
      const newActive = remaining.length ? remaining[0] : null;
      setActiveSession(newActive);
    }
  }

  // send message
  async function handleSend(e) {
    e?.preventDefault();
    if (!input.trim() || !activeSession) return;

    const userMsg = { sender: "user", text: input };

    // optimistic update and persist locally
    setConversations((prev) => {
      const prevMsgs = prev[activeSession] || [];
      const next = { ...prev, [activeSession]: [...prevMsgs, userMsg] };
      saveConversationsToLocal(next);
      return next;
    });

    const payload = {
      user_query: input,
      user_id: userId,
      session_id: activeSession,
    };

    setInput("");
    setLoading(true);

    try {
      // ensure idempotently
      await ensureSessionOnServer(userId, activeSession);

      // call chat
      const res = await axios.post(`${API_URL}/chat`, payload, { timeout: 120000 });

      const botText = (res.data && (res.data.response || res.data.result || res.data.answer)) || "No response";
      const aiMsg = { sender: "bot", text: botText };

      // update state and persist
      setConversations((prev) => {
        const prevMessages = prev[activeSession] || [];
        const next = { ...prev, [activeSession]: [...prevMessages, aiMsg] };
        saveConversationsToLocal(next);
        return next;
      });
    } catch (err) {
      console.error("chat error:", err?.response?.data || err.message);
      const errMsg = { sender: "bot", text: "Sorry — I couldn't process that. Try again." };
      setConversations((prev) => {
        const prevMessages = prev[activeSession] || [];
        const next = { ...prev, [activeSession]: [...prevMessages, errMsg] };
        saveConversationsToLocal(next);
        return next;
      });
    } finally {
      setLoading(false);
    }
  }

  const activeMessages = conversations[activeSession] || [];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h3>Sessions</h3>
          <button className="new-btn" onClick={handleNewChat} disabled={ensuring}>+ New</button>
        </div>

        <div className="session-list">
          {sessions.length === 0 && <div className="empty-note">No sessions yet</div>}
          {sessions.map((sid) => (
            <div
              key={sid}
              className={`session-item ${sid === activeSession ? "active" : ""}`}
              onClick={() => handleSelectSession(sid)}
            >
              <div className="session-title">{sid}</div>
              <div className="session-actions">
                <button
                  className="small"
                  onClick={(e) => { e.stopPropagation(); handleDeleteSession(sid); }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="user-id">User: {userId?.slice(0, 8)}...</div>
        </div>
      </aside>

      <main className="chat-panel">
        <div className="chat-header">
          <h2>{activeSession ? `Session: ${activeSession}` : "No session selected"}</h2>
          {loading && <div className="loading-indicator">Loading…</div>}
        </div>

        <div className="messages">
          {(!activeMessages || activeMessages.length === 0) && !loading && (
            <div className="empty-chat">Start the conversation — say 'hi' </div>
          )}

          {activeMessages.map((m, i) => (
            <div key={i} className={`message-row ${m.sender === "user" ? "user" : "bot"}`}>
              <div className={`bubble ${m.sender === "user" ? "user-bubble" : "bot-bubble"}`}>
                {m.text}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <form className="composer" onSubmit={handleSend}>
          <input
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={!activeSession || loading}
          />
          <button type="submit" disabled={!input.trim() || loading}>Send</button>
        </form>
      </main>
    </div>
  );
}
