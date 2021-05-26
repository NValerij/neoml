/* Copyright © 2021 ABBYY Production LLC

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

#include <common.h>
#pragma hdrstop

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

// class CGatherLayerTest : public CNeoMLTestFixture {
// public:
// 	static bool InitTestFixture() { return true; }
// 	static void DeinitTestFixture() {}
// };

static CPtr<CDnnBlob> packData( IMathEngine& engine, int batchLength, int batchWidth, int channelsCount, const CArray<float>& data )
{
	NeoAssert( batchLength * batchWidth * channelsCount == data.Size() );

	CPtr<CDnnBlob> blob = CDnnBlob::CreateDataBlob(engine, CT_Float, batchLength, batchWidth, channelsCount );
	blob->CopyFrom( data.GetPtr() );

	return blob;
}

TEST(CGatherLayerTest, Forward)
{
	IMathEngine& engine = GetSingleThreadCpuMathEngine();
	CRandom random;
	CDnn dnn( random, engine);

	// Simplest forward only DNN
	auto weights = Source( dnn, "weights" );
	auto indexes = Source( dnn, "indexes" );
	auto gather = Gather()( "gathering", weights, indexes );
	auto sink = Sink( gather, "sink" );

	ASSERT_FALSE( gather->ArePaddingsEnabled() );

	// Data
	weights->SetBlob( packData( engine, 3, 2, 1, {
		1.f,   2.f,
		3.f,   4.f,
		5.f,   6.f
	} ) );
	indexes->SetBlob( packData( engine, 2, 2, 1, {
		0.f,   1.f,
		1.f,   2.f
	} ) );

	// Forward pass
	dnn.RunOnce();

	// Getting results
	auto resultsBlob = sink->GetBlob();
	ASSERT_EQ( 4, resultsBlob->GetDataSize() );

	CArray<float> results;
	results.SetSize( resultsBlob->GetDataSize() );
	resultsBlob->CopyTo( results.GetPtr(), results.Size() );

	// Checking
	const CArray<float> expected = {
		1.f,   4.f,
		3.f,   6.f
	};
	for (int i = 0; i < expected.Size(); i++)
	{
		EXPECT_FLOAT_EQ( expected[i], results[i] ) << "at i = " << i;
	}
}

TEST(CGatherLayerTest, Backward)
{
	IMathEngine& engine = GetSingleThreadCpuMathEngine();
	CRandom random;
	CDnn dnn( random, engine);

	// Solver
	dnn.SetSolver( new CDnnSimpleGradientSolver( engine ) );
	dnn.GetSolver()->SetLearningRate( 1.f );

	// Simplest DNN
	auto input = Source( dnn, "input" );
	auto indexes = Source( dnn, "indexes" );
	// Some trainable layer before Gather-layer
	auto lookup = MultichannelLookup( { CLookupDimension( 4, 1 ) }, true )( "lookup", input );
	auto gather = Gather()( "gathering", lookup, indexes );
	auto etalon = Source( dnn, "etalon" );
	BinaryCrossEntropyLoss()( gather, etalon );

	// Data
	input->SetBlob( packData( engine, 2, 2, 1, {
		0.f, 2.f,
		1.f, 3.f
	} ) );
	lookup->SetEmbeddings( packData( engine, 4, 1, 1, {
		1.f, 2.f, 3.f, 4.f
	} ), 0 );
	// After lookup will be blob:
	// 1.f,   2.f,
	// 3.f,   4.f
	indexes->SetBlob( packData( engine, 1, 2, 1, {
		1.f, // blob[0][1] -> 3.f
		0.f  // lookup[1][0] -> 2.f
	} ) );
	etalon->SetBlob( packData( engine, 1, 2, 1, {
		-3.f,
		-2.f
	} ) );

	// Forward pass
	dnn.RunAndLearnOnce();

	// Getting updated weights
	auto resultsBlob = lookup->GetEmbeddings( 0 );
	ASSERT_EQ( 4, resultsBlob->GetDataSize() );

	CArray<float> results;
	results.SetSize( resultsBlob->GetDataSize() );
	resultsBlob->CopyTo( results.GetPtr(), results.Size() );

	// Checking
	EXPECT_FLOAT_EQ( 1.f, results[0] ); // no update
	EXPECT_NE( 2.f, results[1] );
	EXPECT_NE( 3.f, results[2] );
	EXPECT_FLOAT_EQ( 4.f, results[3] ); // no update
}

TEST(CGatherLayerTest, PaddedForward)
{
	IMathEngine& engine = GetSingleThreadCpuMathEngine();
	CRandom random;
	CDnn dnn( random, engine);

	// Simplest forward only DNN
	auto weights = Source( dnn, "weights" );
	auto indexes = Source( dnn, "indexes" );
	auto gather = Gather()( "gathering", weights, indexes );
	auto sink = Sink( gather, "sink" );

	gather->EnablePaddings();
	ASSERT_TRUE( gather->ArePaddingsEnabled() );

	// Data
	weights->SetBlob( packData( engine, 3, 2, 1, {
		1.f,   2.f,
		3.f,   4.f,
		5.f,   6.f
	} ) );
	indexes->SetBlob( packData( engine, 2, 2, 1, {
		0.f,   1.f,
		-1.f,  0.f
	} ) );

	// Forward pass
	dnn.RunOnce();

	// Getting results
	auto resultsBlob = sink->GetBlob();
	ASSERT_EQ( 4, resultsBlob->GetDataSize() );

	CArray<float> results;
	results.SetSize( resultsBlob->GetDataSize() );
	resultsBlob->CopyTo( results.GetPtr(), results.Size() );

	// Checking
	const CArray<float> expected = {
		1.f,   4.f,
		0.f,   2.f
	};
	for (int i = 0; i < expected.Size(); i++)
	{
		EXPECT_FLOAT_EQ( expected[i], results[i] ) << "at i = " << i;
	}
}

TEST(CGatherLayerTest, PaddedBackward)
{
	IMathEngine& engine = GetSingleThreadCpuMathEngine();
	CRandom random;
	CDnn dnn( random, engine);

	// Solver
	dnn.SetSolver( new CDnnSimpleGradientSolver( engine ) );
	dnn.GetSolver()->SetLearningRate( 1.f );

	// Simplest DNN
	auto input = Source( dnn, "input" );
	auto indexes = Source( dnn, "indexes" );
	// Some trainable layer before Gather-layer
	auto lookup = MultichannelLookup( { CLookupDimension( 4, 1 ) }, true )( "lookup", input );
	auto gather = Gather()( "gathering", lookup, indexes );
	auto etalon = Source( dnn, "etalon" );
	BinaryCrossEntropyLoss()( gather, etalon );

	gather->EnablePaddings();
	ASSERT_TRUE( gather->ArePaddingsEnabled() );

	// Data
	input->SetBlob( packData( engine, 2, 2, 1, {
		0.f,   2.f,
		1.f,   3.f
	} ) );
	lookup->SetEmbeddings( packData( engine, 4, 1, 1, {
		1.f, 2.f, 3.f, 4.f
	} ), 0 );
	// After lookup will be blob:
	// 1.f,   3.f,
	// 2.f,   4.f
	indexes->SetBlob( packData( engine, 2, 2, 1, {
		1.f,   -1.f, // blob[0][1] -> 2.f
		-1.f,   0.f  // blob[1][0] -> 3.f
	} ) );
	etalon->SetBlob( packData( engine, 2, 2, 1, {
		-2.f,   0.f,
		0.f,   -3.f
	} ) );

	// Forward pass
	dnn.RunAndLearnOnce();

	// Getting updated weights
	auto resultsBlob = lookup->GetEmbeddings( 0 );
	ASSERT_EQ( 4, resultsBlob->GetDataSize() );

	CArray<float> results;
	results.SetSize( resultsBlob->GetDataSize() );
	resultsBlob->CopyTo( results.GetPtr(), results.Size() );

	// Checking
	EXPECT_FLOAT_EQ( 1.f, results[0] ); // no update
	EXPECT_NE( 2.f, results[1] );
	EXPECT_NE( 3.f, results[2] );
	EXPECT_FLOAT_EQ( 4.f, results[3] ); // no update
}
