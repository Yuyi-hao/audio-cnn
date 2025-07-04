"use client";
import { useEffect, useState } from "react";
import { Button } from "../../components/ui/button.tsx";
import { Badge } from "../../components/ui/badge.tsx";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card.tsx";
import { Progress } from "../../components/ui/progress.tsx"
import FeatureMap from "../../components/FeatureMap.tsx";
import ColorScale from "../../components/ColorScale.tsx";
import Waveform from "../../components/Waveform.tsx";

interface Prediction{
    class: string;
    confidence: number;
}

interface LayerData{
    shape: number[];
    values: number[][];
}

interface VisualizationData{
    [layerName: string]: LayerData;
}

interface WaveformData{
    values: number[];
    sample_rate: number;
    duration: number;
}

interface APIResponse{
    predictions: Prediction[];
    visualization: VisualizationData;
    input_spectrogram: LayerData;
    waveform: WaveformData;
}

const ESC50_EMOJI_MAP: Record<string, string> = {
  dog: "🐕",
  rain: "🌧️",
  crying_baby: "👶",
  door_wood_knock: "🚪",
  helicopter: "🚁",
  rooster: "🐓",
  sea_waves: "🌊",
  sneezing: "🤧",
  mouse_click: "🖱️",
  chainsaw: "🪚",
  pig: "🐷",
  crackling_fire: "🔥",
  clapping: "👏",
  keyboard_typing: "⌨️",
  siren: "🚨",
  cow: "🐄",
  crickets: "🦗",
  breathing: "💨",
  door_wood_creaks: "🚪",
  car_horn: "📯",
  frog: "🐸",
  chirping_birds: "🐦",
  coughing: "😷",
  can_opening: "🥫",
  engine: "🚗",
  cat: "🐱",
  water_drops: "💧",
  footsteps: "👣",
  washing_machine: "🧺",
  train: "🚂",
  hen: "🐔",
  wind: "💨",
  laughing: "😂",
  vacuum_cleaner: "🧹",
  church_bells: "🔔",
  insects: "🦟",
  pouring_water: "🚰",
  brushing_teeth: "🪥",
  clock_alarm: "⏰",
  airplane: "✈️",
  sheep: "🐑",
  toilet_flush: "🚽",
  snoring: "😴",
  clock_tick: "⏱️",
  fireworks: "🎆",
  crow: "🐦‍⬛",
  thunderstorm: "⛈️",
  drinking_sipping: "🥤",
  glass_breaking: "🔨",
  hand_saw: "🪚",
};

const getEmojiForClass = (className: string): string => {
  return ESC50_EMOJI_MAP[className] || "🔈";
};


function splitLayers(visualization: VisualizationData){
    const main: [string, LayerData][] = [];
    const internals: Record<string, [String, LayerData][]> = {}
    for (const [name, data] of Object.entries(visualization)){
        if(!name.includes(".")){
            main.push([name, data]);
        }else{
            const [parent] = name.split(".");
            if(parent === undefined) continue;
            if(!internals[parent]) internals[parent] = [];
            internals[parent].push([name, data]);
        }
    }

    return {main, internals};
}

const HomePage = () => {
    const [vizData, setVizData] = useState<APIResponse|null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [fileName, setFileName] = useState("");
    const [error, setError] = useState<string|null>(null);
    const [audioURL, setAudioURL] = useState<string | null>(null);


    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event?.target.files?.[0];
        if(!file) return;
        const localAudioFileUrl = URL.createObjectURL(file);
        setAudioURL(localAudioFileUrl);
        setFileName(file.name);
        setIsLoading(true);
        setError(null);
        setVizData(null);

        const reader = new FileReader();
        reader.readAsArrayBuffer(file);
        reader.onload = async () => {
            const api_url = "http://127.0.0.1:5000/predict/";
            try{
                const arrayBuffer = reader.result as ArrayBuffer;
                const base64String = btoa(new Uint8Array(arrayBuffer).reduce((data, byte) => {return data+String.fromCharCode(byte)}, ""));
                const response = await fetch(api_url, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({audio_data: base64String})
                });
                if(!response.ok){
                    throw new Error(`API error: ${response.statusText}`);
                }
                const data:APIResponse = await response.json();
                setVizData(data);
            }
            catch(err){
                setError(err instanceof Error? err.message: "An Unknown error occurred");
            }
            finally{
                setIsLoading(false);
            }
        };
        reader.onerror = () => {
            setError("Failed to read file");
            setIsLoading(false);
        }
    };
    useEffect(() => {
        return () => {
            if (audioURL) {
            URL.revokeObjectURL(audioURL);
            }
        };
    }, [audioURL]);
    const {main, internals} = vizData?splitLayers(vizData?.visualization):{main:[], internals: {}};

    return <main className="min-h-screen bg-stone-50 p-8">
        <div className="mx-auto max-w-[60%]">
            <div className="mb-12 text-center">
                <h1 className="mb-4 text-4xl font-light tracking-light text-grey-900">
                    Audio CNN
                </h1>
                <p className="text-md mb-8 text-stone-600">
                    Upload a WAV file to check if its dog sound or cat sound
                </p>
                <div className="flex flex-col items-center">
                    <div className="relative inline-block">
                        <input type="file" accept=".wav" id="file-upload" disabled={isLoading} className="absolute inset-0 w-full cursor-pointer opacity-0" onChange={handleFileChange}/>
                        <Button variant="outline" size="lg" className="border-stone-100" disabled={isLoading}>{isLoading?"Analyzing...": "Choose a file"}</Button>
                    </div>
                    {fileName && (<>
                        <Badge variant="secondary" className="mt-4 bg-stone-200 text-stone-800">{fileName}</Badge>
                        <audio controls src={audioURL??""} className="mt-2" />
                    </>)}
                </div>
            </div>
            {error && (<Card className="mb-8 border-red-200 bg-red-300">
                <CardContent>
                    <p className="text-red-600">Error: {error}</p>
                </CardContent>
            </Card>)}

            {vizData &&
            (<div className="space-y-8">

                {/* prediction names  */}
                <Card>
                    <CardHeader>
                        <CardTitle className="text-stone-900">Top Predictions</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-3">
                            {vizData.predictions.slice(0, 3).map((pred, idx) => (
                                <div className="space-y-2" key={pred.class}>
                                    <div className="flex items-center justify-between">
                                        <div className="text-md font-medium text-stone-700">
                                            {getEmojiForClass(pred.class)}{" "}
                                            <span>{pred.class.replaceAll("_", " ")}</span>
                                        </div>
                                        <Badge variant={idx===0?"default":"secondary"}>
                                            {(pred.confidence*100).toFixed(1)}%
                                        </Badge>
                                    </div>
                                    <Progress value={pred.confidence*100} className="h-2"></Progress>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>

                <div className="grid-grid-cols-1 gap-6 lg:grid-cols-2">
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-stone-900">Input Spectrogram</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <FeatureMap
                                data={vizData.input_spectrogram.values}
                                title={`${vizData.input_spectrogram.shape.join(" x ")}`}
                                spectrogram
                            />
                            <div className="mt-5 flex justify-end">
                                <ColorScale width={200} height={16} min={-1} max={1} />
                            </div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader>
                            <CardTitle className="text-stone-900">
                                Audio Waveform
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <Waveform
                                data={vizData.waveform.values}
                                title={`${vizData.waveform.duration.toFixed(2)}s * ${vizData.waveform.sample_rate}Hz`}
                            />
                        </CardContent>
                    </Card>
                </div>

                {/* Feature maps */}
                <Card>
                    <CardHeader>
                        <CardTitle>Convolution Layer Outputs</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-5 gap-6">
                            {main.map(([mainName, mainData]) => (
                                <div className="space-y-4" key={mainName}>
                                    <div>
                                        <h4 className="mb-2 font-medium text-stone-700">{mainName}</h4>
                                        <FeatureMap
                                            data={mainData.values}
                                            title={`${mainData.shape.join(" x ")}`}
                                        />
                                    </div>

                                    {internals[mainName] && (
                                        <div className="h-80 overflow-y-auto rounded border border-stone-200 bg-stone-50 p-2">
                                            <div className="space-y-2">
                                                {internals[mainName]
                                                .sort(([a], [b]) => a.toString().localeCompare(b.toString()))
                                                .map(([layerName, layerData]) => (
                                                    <FeatureMap
                                                        key={layerName.toString()}
                                                        data={layerData.values}
                                                        title={layerName.replace(`${mainName}.`, "")}
                                                        internal={true}
                                                    />
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                </div>
                            ))}
                        </div>
                        <div className="mt-5 flex justify-end">
                            <ColorScale width={200} height={16} min={-1} max={1} />
                        </div>
                    </CardContent>
                </Card>

            </div>)}
        </div>
    </main>
}

export default HomePage;