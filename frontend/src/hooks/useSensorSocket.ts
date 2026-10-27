import { useEffect, useRef, useState } from "react";
import { io, type Socket } from "socket.io-client";
import type { SensorUpdate } from "../types";

export interface SensorSocketState {
  connected: boolean;
  latest: SensorUpdate | null;
}

export function useSensorSocket(): SensorSocketState {
  const [connected, setConnected] = useState(false);
  const [latest, setLatest] = useState<SensorUpdate | null>(null);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    const socket: Socket = io(window.location.origin, {
      transports: ["websocket", "polling"],
    });
    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));
    socket.on("sensor_update", (data: SensorUpdate) => setLatest(data));

    return () => {
      socket.off("connect");
      socket.off("disconnect");
      socket.off("sensor_update");
      socket.disconnect();
      socketRef.current = null;
    };
  }, []);

  return { connected, latest };
}
