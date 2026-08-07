import { useState } from "react";

import { Badge, Button, Empty, Input, PageHeader, Panel } from "../components/ui";
import type { Tone } from "../components/ui";
import { useEcho } from "../hooks/useEcho";

const STATE_LABEL: Record<string, { text: string; tone: Tone }> = {
  connecting: { text: "Connecting", tone: "warning" },
  open: { text: "Connected", tone: "normal" },
  closed: { text: "Disconnected", tone: "defect" },
};

export function EchoRoute() {
  const { state, received, send } = useEcho();
  const [draft, setDraft] = useState("");

  const badge = STATE_LABEL[state] ?? { text: state, tone: "neutral" as Tone };

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="WebSocket echo"
        actions={<Badge tone={badge.tone}>{badge.text}</Badge>}
        meta={
          <span>
            Proves the WebSocket path end to end. Frames use the same{" "}
            <code className="font-mono text-fg">ev</code> envelope the job stream uses.
          </span>
        }
      />

      <Panel title="Send a frame">
        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (draft.trim() === "") return;
            send(draft);
            setDraft("");
          }}
        >
          <Input
            aria-label="Message to echo"
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Type a message"
            value={draft}
          />
          <Button variant="primary" type="submit" disabled={state !== "open"}>
            Send
          </Button>
        </form>
      </Panel>

      <Panel title={`Received (${received.length})`}>
        {received.length === 0 ? (
          <Empty>No frames received yet.</Empty>
        ) : (
          <ul className="flex flex-col gap-1">
            {received.map((event, index) => (
              <li
                key={index}
                className="rounded-control border border-line px-3 py-2 font-mono text-sm text-fg"
              >
                <span className="text-fg-subtle">{event.ev}</span> {event.message}
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
