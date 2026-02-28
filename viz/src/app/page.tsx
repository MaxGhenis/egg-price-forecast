import data from "@/data/forecast.json";
import { Dashboard } from "@/components/Dashboard";
import type { ForecastData } from "@/lib/types";

export default function Page() {
  return <Dashboard data={data as ForecastData} />;
}
