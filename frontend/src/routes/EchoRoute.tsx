import { useState } from "react";

import { useEcho } from "../hooks/useEcho";

const STATE_LABEL = {
  connecting: { text: "Connecting", className: "bg-amber-100 text-amber-800" },
  open: { text: "Connected", className: "bg-emerald-100 text-emerald-800" },
  closed: { text: "Disconnected", className: "bg-red-100 text-red-800" },
} as const;

export function EchoRoute() {
  const { state, received, send } = useEcho();
  const [draft, setDraft] = useState("");

  const badge = STATE_LABEL[state];

  return (
    <section className="space-y-4">
      <header className="flex items-baseline gap-3">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">WebSocket echo</h2>
        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${badge.className}`}>
          {badge.text}
        </span>
      </header>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Proves the WebSocket path end to end. Frames use the same <code>ev</code> envelope the
        job stream will use.
      </p>

      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (draft.trim() === "") return;
          send(draft);
          setDraft("");
        }}
      >
        <input
          aria-label="Message to echo"
          className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Type a message"
          value={draft}
        />
        <button
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40 dark:bg-slate-100 dark:text-slate-900"
          disabled={state !== "open"}
          type="submit"
        >
          Send
        </button>
      </form>

      {received.length === 0 ? (
        <p className="text-sm text-slate-500">No frames received yet.</p>
      ) : (
        <ul className="space-y-1">
          {received.map((event, index) => (
            <li
              key={index}
              className="rounded-md border border-slate-200 px-3 py-2 font-mono text-sm dark:border-slate-700"
            >
              <span className="text-slate-400">{event.ev}</span> {event.message}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
