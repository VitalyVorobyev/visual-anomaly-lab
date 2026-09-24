import { useEffect, useState } from "react";

/** An `<img>` decoded off-DOM, for Konva. `null` until it has loaded, or when it failed. */
export function useHtmlImage(
  src: string | undefined,
  crossOrigin?: "anonymous",
): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    setImage(null);
    if (!src) return;
    const next = new globalThis.Image();
    if (crossOrigin) next.crossOrigin = crossOrigin;
    next.onload = () => setImage(next);
    next.onerror = () => setImage(null);
    next.src = src;
    return () => {
      next.onload = null;
      next.onerror = null;
    };
  }, [src, crossOrigin]);
  return image;
}
