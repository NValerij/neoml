/* Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Layer that pickups particular elements from input (based on user given indexes).
// It is possible to enable paddings for case of different length sequences.
class NEOML_API CGatherLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CGatherLayer )
public:
    // Layer inputs
    enum TInput {
        I_Weights, // float weights to select from (batch length, batch width, 1, *)
        I_Indexes, // (float) sample indexes to select (batch length', batch width, 1, 1, 1, 1, 1)

        I_Count
    };
    // Layer output: selected weights (batch length', batch width, 1, *)

	explicit CGatherLayer( IMathEngine& mathEngine );

    // Control whether use paddings or not (it has additional performance cost).
    // Padding is an index with value -1.
    void EnablePaddings( bool enable = true ) { arePaddingsUsed = enable; }
    bool ArePaddingsEnabled() const { return arePaddingsUsed; }

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
    bool arePaddingsUsed = false;

    void addOptionalPaddings( CPtr<CDnnBlob>& weights, CPtr<const CDnnBlob>& indexes ) const;
    void flatternIndexes( const CDnnBlob& indexes, CFloatHandleStackVar& result ) const;
};

NEOML_API CLayerWrapper<CGatherLayer> Gather();

} // namespace NeoML
