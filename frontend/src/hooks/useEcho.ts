import { useCallback, useEffect, useRef, useState } from "react";

import { websocketUrl } from "../api/baseUrl";

export type ConnectionState = "connecting" | "open" | "closed";

export interface EchoEvent {
  ev: string;
  message: string;
}

/**
 * The WebSocket half of the walking skeleton.
 *
 * Deliberately kept close to the shape M3 will need for `/ws/jobs/{id}`: one socket held
 * for the lifetime of the component, a connection state the UI can render, and frames
 * parsed as ADR-0009 events rather than raw strings.
 */
export function useEcho() {
  const [state, setState] = useState<ConnectionState>("connecting");
  const [received, setReceived] = useState<EchoEvent[]>([]);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket(websocketUrl("/ws/echo"));
    socketRef.current = socket;

    socket.onopen = () => setState("open");
    socket.onclose = () => setState("closed");
    socket.onerror = () => setState("closed");
    socket.onmessage = (event: MessageEvent<string>) => {
      setReceived((previous) => [...previous, JSON.parse(event.data) as EchoEvent]);
    };

    return () => {
      socketRef.current = null;
      socket.close();
    };
  }, []);

  const send = useCallback((message: string) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(message);
    }
  }, []);

  return { state, received, send };
}
